import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from playwright.async_api import async_playwright
from .extraction import extract, inspect


class Policy:
    def __init__(self, hosts: set[str]):
        self.hosts = hosts

    async def validate(self, url: str):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError("Unsupported URL")
        if parsed.hostname not in self.hosts or parsed.port not in {None, 80, 443}:
            raise ValueError("Destination outside configured browsing scope")
        addresses = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, parsed.port or 443)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError("Non-public destination denied")


async def scan_page(url: str, policy: Policy) -> dict:
    await policy.validate(url)
    raw_documents = []
    blocked = []
    # All browser HTTP requests are fulfilled through this bounded transport.
    async with httpx.AsyncClient(timeout=15, follow_redirects=False, trust_env=False) as transport:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                context = await browser.new_context(service_workers="block", accept_downloads=False)

                async def route_request(route):
                    request = route.request
                    try:
                        await policy.validate(request.url)
                        if request.method != "GET":
                            raise ValueError("Only GET is permitted")
                        async with transport.stream("GET", request.url) as response:
                            body = bytearray()
                            async for chunk in response.aiter_bytes():
                                body.extend(chunk)
                                if len(body) > 2_000_000:
                                    raise ValueError("Response exceeds size limit")
                            if request.is_navigation_request() and request.frame == context.pages[0].main_frame:
                                if "text/html" in response.headers.get("content-type", ""):
                                    raw_documents.append((request.url, bytes(body).decode(response.encoding or "utf-8", errors="replace")))
                            headers = {key: value for key, value in response.headers.items()
                                       if key.lower() not in {"content-encoding", "content-length", "transfer-encoding", "set-cookie"}}
                            await route.fulfill(status=response.status_code, headers=headers, body=bytes(body))
                    except Exception:
                        blocked.append(request.url)
                        await route.abort()

                await context.route("**/*", route_request)
                await context.route_web_socket("**/*", lambda route: route.close())
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=25000)
                await page.wait_for_timeout(750)
                # Mark computed CSS visibility in the snapshot for extraction.
                await page.evaluate("""() => { for (const e of document.querySelectorAll('*')) {
                    const s = getComputedStyle(e);
                    if (s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') e.setAttribute('hidden', '');
                }}""")
                rendered = await page.content()
                if len(rendered) > 2_000_000:
                    raise ValueError("Rendered document exceeds size limit")
                passages = []
                for source_url, html in raw_documents:
                    passages.extend(extract(html, source_url, "raw"))
                passages.extend(extract(rendered, page.url, "rendered"))
                return {"id": str(uuid4()), "url": page.url, **inspect(passages[:2000]),
                        "coverage": {"partial": bool(blocked) or len(passages) > 2000,
                                     "blocked_requests": len(blocked), "snapshot_delay_ms": 750,
                                     "limitations": ["Main document only", "No closed shadow DOM", "Bounded dynamic snapshot"]}}
            finally:
                await browser.close()
