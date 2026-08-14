"""Local live-preview server shared by `clap-teleop` and `clap-rollout-deploy`: serves a
viewer HTML page over HTTP and streams predicted frames (+ arbitrary per-frame metadata)
to it over a websocket.

Both servers run in background threads, so they don't interfere with
`clap.rollout.teleop`'s blocking raw-terminal keypress reads (or `clap.rollout.deploy`'s
policy calls) on the main thread. Serving the HTML over HTTP (not opening it as a local
file) matters: a file:// path only resolves on whatever machine has that path on disk --
useless to a browser on your laptop when the session runs on a remote/SLURM compute node.
`start()` prints the right ssh -L command for either case -- a SLURM compute node is a
separate machine from the login node you ssh into, so its tunnel needs to name the
compute node explicitly (not "localhost") as the forward's remote endpoint. No
GUI/X11/DISPLAY dependency either way.
"""

import asyncio
import base64
import functools
import http.server
import json
import logging
import os
import socket
import threading

import cv2

logger = logging.getLogger(__name__)


class LiveViewServer:
    """Broadcasts JPEG-encoded frames (+ arbitrary JSON-able metadata) to every browser tab
    connected over ws://host:ws_port, and serves the viewer page itself over http://host:http_port.

    `broadcast_frame` is the only method meant to be called from the main
    (synchronous) thread -- it hands the frame to the background event loop
    via `run_coroutine_threadsafe` rather than touching asyncio state directly.

    Args:
        viewer_page: Filename (under examples/getting_started/) of the HTML page to
            serve and link to -- e.g. "teleop_viewer.html" or "deploy_viewer.html".
    """

    def __init__(self, host=None, ws_port=8765, http_port=8766, viewer_page="teleop_viewer.html"):
        # Binding "localhost" (127.0.0.1) only accepts connections arriving via loopback --
        # fine for a plain local session, but a `ssh -L port:<compute-node>:port <login-node>`
        # tunnel has the LOGIN node dial the compute node's real network interface, not its
        # loopback, so a loopback-only bind silently refuses it. Default to 0.0.0.0 (all
        # interfaces) under SLURM so that already-standard forwarding pattern just works.
        self.host = host or ("0.0.0.0" if os.environ.get("SLURM_JOB_ID") else "localhost")
        self.ws_port = ws_port
        self.http_port = http_port
        self.viewer_page = viewer_page
        self._clients = set()
        self._loop = None
        self._ws_server = None
        self._ws_thread = None
        self._http_server = None
        self._http_thread = None
        self._seed_payload = None  # the one-time seed/initial-frame payload -- its own slot on the page
        self._last_payload = None  # most recent live-prediction frame payload
        self._last_meta_payload = None  # most recent metadata-only payload (e.g. a mode toggle, no new frame)

    def has_clients(self):
        """Whether any browser tab is currently connected -- lets a caller skip preparing
        (decoding/encoding) frames nobody would actually see, e.g. multi-frame-per-round
        broadcasts that would otherwise cost real GPU decode time even with zero viewers."""
        return bool(self._clients)

    def start(self, timeout=10):
        """Launch both servers in background threads; blocks until the websocket one is accepting connections."""
        import websockets

        ready = threading.Event()

        async def handler(websocket):
            self._clients.add(websocket)
            logger.info(f"✅ live view: browser connected ({websocket.remote_address}), {len(self._clients)} client(s) now")
            # Replay the seed frame + most recent live-prediction frame immediately -- otherwise
            # a tab that connects after either was already broadcast (the common case: the
            # session usually starts before you've even opened the browser) would just sit on
            # the "waiting..." placeholder until the NEXT live-prediction frame comes in.
            if self._seed_payload is not None:
                await websocket.send(self._seed_payload)
            if self._last_payload is not None:
                await websocket.send(self._last_payload)
            if self._last_meta_payload is not None:
                # Sent last so a mode toggle after the most recent frame (the common case --
                # Tab/Space don't produce a new frame) isn't clobbered by _last_payload above.
                await websocket.send(self._last_meta_payload)
            try:
                await websocket.wait_closed()  # keep the connection open until the browser tab disconnects
            finally:
                self._clients.discard(websocket)
                logger.info(f"🔌 live view: browser disconnected, {len(self._clients)} client(s) left")

        async def _serve():
            self._ws_server = await websockets.serve(handler, self.host, self.ws_port)
            ready.set()  # signal the main thread that start() can return
            await self._ws_server.wait_closed()

        def _run_ws_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(_serve())

        self._ws_thread = threading.Thread(target=_run_ws_loop, daemon=True)  # daemon: don't block process exit
        self._ws_thread.start()
        if not ready.wait(timeout=timeout):
            raise RuntimeError(f"live-view websocket server didn't start within {timeout}s")

        # Static file server rooted at examples/getting_started/ (where teleop_viewer.html
        # lives) -- assumes cwd is the repo root, same as every other getting_started/*.sh.
        viewer_dir = os.path.abspath("examples/getting_started")
        handler_cls = functools.partial(http.server.SimpleHTTPRequestHandler, directory=viewer_dir)
        self._http_server = http.server.ThreadingHTTPServer((self.host, self.http_port), handler_cls)
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()

        # Always "localhost" in the printed URL, regardless of what self.host is bound to --
        # that's what the browser (via a direct connection or a tunnel's client-side end)
        # always targets; self.host only controls which server-side interface accepts it.
        url = f"http://localhost:{self.http_port}/{self.viewer_page}?port={self.ws_port}"
        logger.info(f"🔌 live view: ws://{self.host}:{self.ws_port} (frames) + http://{self.host}:{self.http_port} (viewer page)")
        logger.info(f"🔗 open in a browser: {url}")
        if os.environ.get("SLURM_JOB_ID"):
            # A SLURM compute node is a SEPARATE machine from the login node you actually ssh
            # into -- its "localhost" isn't the login node's, so `-L port:localhost:port` alone
            # only forwards to the login node's own (unused) loopback. The tunnel's remote
            # endpoint needs to name this compute node explicitly instead.
            node = socket.gethostname()
            logger.info(f"   SLURM job detected (node={node}) -- two-hop tunnel needed, run this "
                        f"from your LOCAL machine (replace <login-node> with the host you ssh into):")
            logger.info(f"     ssh -L {self.ws_port}:{node}:{self.ws_port} -L {self.http_port}:{node}:{self.http_port} <login-node>")
        else:
            logger.info(f"   remote (non-SLURM) session? port-forward both first: "
                        f"ssh -L {self.ws_port}:localhost:{self.ws_port} -L {self.http_port}:localhost:{self.http_port} <host>")

    def broadcast_frame(self, frame_u8_hwc_rgb, seed=False, **meta):
        """Encode one HWC uint8 RGB frame as base64 JPEG and push it (+ any caller-supplied
        metadata, e.g. key=/round=/instruction=) to every connected client as JSON.

        seed: True for the one-time initial/seed frame (its own slot on the page,
            cached separately and never overwritten by later live-prediction frames)
            vs. False for a regular per-step/per-round live-prediction frame.

        Always encodes and caches (even with zero clients currently connected --
        the session typically starts before you've opened the browser at all) so a
        client connecting later still gets it immediately; only the actual send is
        skipped when nobody's listening yet.
        """
        if self._loop is None:
            return  # server not started
        ok, buf = cv2.imencode(".jpg", frame_u8_hwc_rgb[:, :, ::-1])  # RGB -> BGR, cv2's own convention
        if not ok:
            logger.warning("⚠️ live view: JPEG encode failed, skipping this frame")
            return
        b64 = base64.b64encode(buf).decode("ascii")
        payload = json.dumps({"image": b64, "seed": seed, **meta})
        if seed:
            self._seed_payload = payload
        else:
            self._last_payload = payload
        if self._clients:
            asyncio.run_coroutine_threadsafe(self._send_all(payload), self._loop)

    def broadcast_meta(self, **meta):
        """Push a metadata-only update (no new/changed frame) to every connected client --
        e.g. `clap.rollout.teleop.TeleopSession`'s active_target/dual mode toggles, which
        change no pose and trigger no model call. Cheap: no JPEG encode, unlike
        `broadcast_frame`. Cached the same way (even with zero clients currently connected)
        so a client connecting later still gets the current mode immediately.
        """
        if self._loop is None:
            return  # server not started
        payload = json.dumps({"meta_only": True, **meta})
        self._last_meta_payload = payload
        if self._clients:
            asyncio.run_coroutine_threadsafe(self._send_all(payload), self._loop)

    async def _send_all(self, payload):
        if self._clients:
            await asyncio.gather(*(ws.send(payload) for ws in list(self._clients)), return_exceptions=True)

    def broadcast_frames(self, frames_u8_hwc_rgb, fps=4, **meta):
        """Broadcast a whole batch of live-prediction frames, played out at `fps`.

        Unlike `broadcast_frame`, this returns immediately -- JPEG-encoding, pacing
        (`asyncio.sleep(1/fps)` between frames), and sending all happen inside a single
        task on the background event loop, never on the caller's thread. That matters
        because the caller is typically the main inference loop (clap-teleop's keypress
        loop, clap-rollout-deploy's policy loop): pacing frames there directly would add
        real wall-clock delay to every step/round, which this sidesteps entirely.
        """
        if self._loop is None:
            return  # server not started
        asyncio.run_coroutine_threadsafe(self._send_paced(list(frames_u8_hwc_rgb), fps, meta), self._loop)

    async def _send_paced(self, frames, fps, meta):
        delay = 1.0 / fps if fps > 0 else 0.0
        for i, frame_u8_hwc_rgb in enumerate(frames):
            ok, buf = cv2.imencode(".jpg", frame_u8_hwc_rgb[:, :, ::-1])  # RGB -> BGR, cv2's own convention
            if not ok:
                logger.warning("⚠️ live view: JPEG encode failed, skipping this frame")
                continue
            b64 = base64.b64encode(buf).decode("ascii")
            payload = json.dumps({"image": b64, "seed": False, **meta})
            self._last_payload = payload  # cached even with zero clients, same reasoning as broadcast_frame
            if self._clients:
                await asyncio.gather(*(ws.send(payload) for ws in list(self._clients)), return_exceptions=True)
            if delay and i < len(frames) - 1:
                await asyncio.sleep(delay)  # off the caller's thread -- doesn't touch the main inference loop's timing

    def stop(self):
        """Shut both servers down; safe to call even if they were never started."""
        if self._loop is not None and self._ws_server is not None:
            self._loop.call_soon_threadsafe(self._ws_server.close)
        if self._http_server is not None:
            self._http_server.shutdown()