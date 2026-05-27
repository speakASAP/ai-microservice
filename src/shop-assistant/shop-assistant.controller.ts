import { Controller, Post, Body, HttpCode, HttpStatus } from '@nestjs/common';
import { Public } from '../service-identity/public.decorator';
import { ShopAssistantService } from './shop-assistant.service';

@Controller('api/shop-assistant')
@Public()
export class ShopAssistantController {
  constructor(private readonly shopAssistantService: ShopAssistantService) {}

  @Post('transcribe')
  @HttpCode(HttpStatus.OK)
  transcribe(@Body() body: { voice_file_url: string }) {
    return this.shopAssistantService.transcribe(body.voice_file_url);
  }

  @Post('refine-query')
  @HttpCode(HttpStatus.OK)
  refineQuery(
    @Body()
    body: {
      user_text: string;
      previous_params?: Record<string, unknown>;
      role?: string;
      prompt_content?: string;
      model?: string;
    },
  ) {
    return this.shopAssistantService.refineQuery(
      body.user_text,
      body.previous_params,
      body.role,
      body.prompt_content,
      body.model,
    );
  }

  @Post('search')
  @HttpCode(HttpStatus.OK)
  search(@Body() body: { query_text: string; limit?: number }) {
    return this.shopAssistantService.search(body.query_text, body.limit);
  }

  @Post('format-presentation')
  @HttpCode(HttpStatus.OK)
  formatPresentation(
    @Body()
    body: {
      results: Record<string, unknown>[];
      query_text: string;
      role?: string;
      prompt_content?: string;
      model?: string;
    },
  ) {
    return this.shopAssistantService.formatPresentation(
      body.results,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
    );
  }

  @Post('compare-prices')
  @HttpCode(HttpStatus.OK)
  comparePrices(
    @Body()
    body: {
      results: Record<string, unknown>[];
      query_text: string;
      role?: string;
      prompt_content?: string;
      model?: string;
      priority_order?: string[];
    },
  ) {
    return this.shopAssistantService.comparePrices(
      body.results,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
      body.priority_order,
    );
  }

  @Post('extract-location')
  @HttpCode(HttpStatus.OK)
  extractLocation(
    @Body()
    body: {
      user_text: string;
      query_text: string;
      role?: string;
      prompt_content?: string;
      model?: string;
      priority_order?: string[];
    },
  ) {
    return this.shopAssistantService.extractLocation(
      body.user_text,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
      body.priority_order,
    );
  }
}
