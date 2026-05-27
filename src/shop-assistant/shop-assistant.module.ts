import { Module } from '@nestjs/common';
import { ShopAssistantController } from './shop-assistant.controller';
import { ShopAssistantService } from './shop-assistant.service';
import { AiModule } from '../ai/ai.module';

@Module({
  imports: [AiModule],
  controllers: [ShopAssistantController],
  providers: [ShopAssistantService],
})
export class ShopAssistantModule {}
