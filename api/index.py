from backend.main import app as fastapi_app


class StripApiPrefix:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/api/"):
            scope = {**scope, "path": scope["path"][4:]}
            if "raw_path" in scope and scope["raw_path"].startswith(b"/api/"):
                scope["raw_path"] = scope["raw_path"][4:]
        await self.app(scope, receive, send)


app = StripApiPrefix(fastapi_app)