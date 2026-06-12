import { Controller, Get, NotFoundException, Param, Res } from '@nestjs/common';
import { Response } from 'express';
import { existsSync } from 'fs';
import { join } from 'path';
import { Public } from '../service-identity/public.decorator';

const ADMIN_PUBLIC_DIR = join(process.cwd(), 'public', 'admin');
const CONTENT_TYPES: Record<string, string> = {
  'app.js': 'application/javascript; charset=utf-8',
  'styles.css': 'text/css; charset=utf-8',
};

@Controller()
export class AdminFrontendController {
  @Public()
  @Get(['admin', 'admin/'])
  index(@Res() res: Response) {
    return res.type('html').sendFile(join(ADMIN_PUBLIC_DIR, 'index.html'));
  }

  @Public()
  @Get('admin/assets/:asset')
  asset(@Param('asset') asset: string, @Res() res: Response) {
    if (!/^[a-z0-9.-]+$/.test(asset) || !(asset in CONTENT_TYPES)) {
      throw new NotFoundException();
    }

    const filePath = join(ADMIN_PUBLIC_DIR, asset);
    if (!existsSync(filePath)) throw new NotFoundException();
    return res.type(CONTENT_TYPES[asset]).sendFile(filePath);
  }
}
