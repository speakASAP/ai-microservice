import { Module } from '@nestjs/common';
import { DocumentExtractorService } from './document-extractor.service';
import { DocumentsController } from './documents.controller';

@Module({
  controllers: [DocumentsController],
  providers: [DocumentExtractorService],
  exports: [DocumentExtractorService],
})
export class DocumentsModule {}
