"""Keep the model-server launcher's health-server hook covered without a listener."""

import httpx
from endurance.model_server_launcher import ProbedHealthServer

from reactor_runtime.core import Health, RuntimeState


async def test_probe_route_uses_the_worker_request_count() -> None:
    server = ProbedHealthServer(
        host="127.0.0.1",
        port=0,
        process_health=Health.healthy,
        state=lambda: RuntimeState.AVAILABLE,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server._app), base_url="http://test"
    ) as client:
        response = await client.get("/_endurance/model_server")
        assert response.status_code == 200
        assert response.json() == {"pending_requests": 0}
        assert (await client.get("/health")).json()["state"] == "available"
