from __future__ import annotations

import json
import ssl

import websocket


def main() -> None:
    ws = websocket.create_connection(
        "wss://geofresh.org/websocket/",
        origin="https://geofresh.org",
        timeout=20,
        sslopt={"cert_reqs": ssl.CERT_REQUIRED},
    )
    ws.settimeout(10)
    ws.send(
        json.dumps(
            {
                "method": "init",
                "data": {
                    "singletons": "",
                    "allowDataUriScheme": True,
                    ".clientdata_pixelratio": 1,
                    ".clientdata_url_protocol": "https:",
                    ".clientdata_url_hostname": "geofresh.org",
                    ".clientdata_url_pathname": "/",
                    ".clientdata_single_pixel_ratio": 1,
                },
            }
        )
    )
    for _ in range(20):
        try:
            message = ws.recv()
        except TimeoutError:
            print("TIMEOUT")
            break
        if isinstance(message, bytes):
            print("BINARY", len(message))
        else:
            print(message[:10_000])
    ws.close()


if __name__ == "__main__":
    main()
