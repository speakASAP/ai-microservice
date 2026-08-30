import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import type { NestExpressApplication } from '@nestjs/platform-express';
import { MAX_DOCUMENT_BYTES } from './contracts';
import { AppModule } from './app.module';
import { ContractViolationFilter } from './common/filters/contract-violation.filter';

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);

  // /documents/extract carries a base64 document, and base64 inflates by ~4/3. Express
  // defaults to 100kb, which would reject every real PDF with a bare 413 that reads as a
  // gateway fault rather than a body-limit one.
  app.useBodyParser('json', { limit: Math.ceil((MAX_DOCUMENT_BYTES * 4) / 3) + 1024 * 1024 });

  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
    }),
  );

  app.useGlobalFilters(new ContractViolationFilter());

  const port = process.env.PORT || 3380;
  await app.listen(port);
  console.log(`AI Microservice listening on port ${port}`);
}

bootstrap().catch((err) => {
  console.error('Failed to bootstrap application:', err);
  process.exit(1);
});
