"""SSRF protection for any tool that fetches a URL (agents are a perfect SSRF proxy)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class UnsafeURL(ValueError):
    pass


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def validate_url(url: str, allowed_domains: list[str], *, resolve: bool = True) -> str:
    """Return the URL if safe; raise UnsafeURL otherwise.

    https only, hostname on the allowlist, no credentials in URL, no private/link-local IPs
    (checked on literal IPs AND on DNS results to blunt DNS-rebinding style tricks).
    """
    parts = urlparse(url)
    if parts.scheme != "https":
        raise UnsafeURL("https required")
    if parts.username or parts.password:
        raise UnsafeURL("credentials in URL not allowed")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURL("missing host")
    try:
        ipaddress.ip_address(host)
        is_ip_literal = True
    except ValueError:
        is_ip_literal = False
    if is_ip_literal:
        raise UnsafeURL("raw IP addresses not allowed")
    if not any(host == d or host.endswith("." + d) for d in (x.lower() for x in allowed_domains)):
        raise UnsafeURL("domain not on allowlist")
    if resolve:
        try:
            infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise UnsafeURL("cannot resolve host") from exc
        if not infos or not all(_is_public(i[4][0]) for i in infos):
            raise UnsafeURL("resolves to non-public address")
    return url
