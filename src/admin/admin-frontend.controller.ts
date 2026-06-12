import { Body, Controller, Get, NotFoundException, Param, Post, Req, Res, UnauthorizedException } from '@nestjs/common';
import { Request, Response } from 'express';
import { existsSync } from 'fs';
import { join } from 'path';
import { Public } from '../service-identity/public.decorator';
import { AdminAuthService, ADMIN_SESSION_COOKIE } from './admin-auth.service';

const ADMIN_PUBLIC_DIR = join(process.cwd(), 'public', 'admin');
const CONTENT_TYPES: Record<string, string> = {
  'app.js': 'application/javascript; charset=utf-8',
  'styles.css': 'text/css; charset=utf-8',
};

@Controller()
export class AdminFrontendController {
  constructor(private readonly adminAuth: AdminAuthService) {}

  @Public()
  @Get(['admin', 'admin/'])
  async index(@Req() req: Request, @Res() res: Response) {
    try {
      await this.adminAuth.requireAdminFromRequest(req);
    } catch (_err) {
      return res.redirect(302, this.adminAuth.getAuthLoginUrl());
    }

    return res.type('html').sendFile(join(ADMIN_PUBLIC_DIR, 'index.html'));
  }

  @Public()
  @Get('admin/assets/:asset')
  async asset(@Param('asset') asset: string, @Req() req: Request, @Res() res: Response) {
    try {
      await this.adminAuth.requireAdminFromRequest(req);
    } catch (_err) {
      throw new UnauthorizedException('Admin authentication required');
    }

    if (!/^[a-z0-9.-]+$/.test(asset) || !(asset in CONTENT_TYPES)) {
      throw new NotFoundException();
    }

    const filePath = join(ADMIN_PUBLIC_DIR, asset);
    if (!existsSync(filePath)) throw new NotFoundException();
    return res.type(CONTENT_TYPES[asset]).sendFile(filePath);
  }

  @Public()
  @Get('admin/session')
  sessionBridge(@Res() res: Response) {
    return res.type('html').send(this.sessionBridgeHtml());
  }

  @Public()
  @Post('admin/session')
  async createSession(@Body('accessToken') accessToken: string, @Res() res: Response) {
    if (!accessToken) throw new UnauthorizedException('Missing access token');

    await this.adminAuth.requireAdminToken(accessToken);
    const maxAge = this.adminAuth.getCookieMaxAgeMs(accessToken);
    if (maxAge <= 0) throw new UnauthorizedException('Expired access token');

    res.cookie(ADMIN_SESSION_COOKIE, accessToken, {
      httpOnly: true,
      secure: true,
      sameSite: 'lax',
      path: '/admin',
      maxAge,
    });
    return res.status(204).send();
  }

  @Public()
  @Get('admin/logout')
  logout(@Res() res: Response) {
    res.clearCookie(ADMIN_SESSION_COOKIE, {
      httpOnly: true,
      secure: true,
      sameSite: 'lax',
      path: '/admin',
    });
    return res.redirect(302, this.adminAuth.getAuthLoginUrl());
  }

  private sessionBridgeHtml(): string {
    return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI Admin Authentication</title>
    <style>
      body { margin: 0; min-height: 100vh; display: grid; place-items: center; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f7f8fb; }
      main { width: min(420px, calc(100vw - 32px)); padding: 24px; border: 1px solid #d8deea; border-radius: 8px; background: #fff; box-shadow: 0 16px 40px rgba(23,32,51,.08); }
      h1 { margin: 0 0 8px; font-size: 20px; }
      p { margin: 0; color: #5f6b7a; line-height: 1.45; }
      .error { color: #b42318; }
    </style>
  </head>
  <body>
    <main>
      <h1>Checking admin access</h1>
      <p id="message">Completing authentication...</p>
    </main>
    <script>
      (async function () {
        const message = document.getElementById("message");
        const params = new URLSearchParams(window.location.hash.slice(1));
        const accessToken = params.get("access_token");
        if (!accessToken) {
          window.location.replace("/admin/logout");
          return;
        }
        try {
          const response = await fetch("/admin/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ accessToken }),
            credentials: "same-origin"
          });
          if (!response.ok) throw new Error("Admin access denied");
          window.history.replaceState(null, "", "/admin/session");
          window.location.replace("/admin");
        } catch (error) {
          message.textContent = error.message || "Admin access denied";
          message.className = "error";
        }
      })();
    </script>
  </body>
</html>`;
  }
}
