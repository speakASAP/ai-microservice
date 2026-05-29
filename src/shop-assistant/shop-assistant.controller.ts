import { Controller, Post, Body, HttpCode, HttpStatus, UsePipes } from '@nestjs/common';
import { Public } from '../service-identity/public.decorator';
import { ShopAssistantService } from './shop-assistant.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow } from '../contracts/parse-or-throw';
import {
  ShopTranscribeRequestSchema,
  ShopRefineQueryRequestSchema,
  ShopSearchRequestSchema,
  ShopPresentationRequestSchema,
  ShopComparePricesRequestSchema,
  ShopExtractLocationRequestSchema,
  ShopTranscribeResponseSchema,
  ShopRefineQueryResponseSchema,
  ShopSearchResponseSchema,
  ShopPresentationResponseSchema,
  ShopComparePricesResponseSchema,
  ShopExtractLocationResponseSchema,
} from '../contracts';
import type {
  ShopTranscribeRequest,
  ShopRefineQueryRequest,
  ShopSearchRequest,
  ShopPresentationRequest,
  ShopComparePricesRequest,
  ShopExtractLocationRequest,
} from '../contracts';

@Controller('api/shop-assistant')
@Public()
export class ShopAssistantController {
  constructor(private readonly shopAssistantService: ShopAssistantService) {}

  @Post('transcribe')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopTranscribeRequestSchema))
  async transcribe(@Body() body: ShopTranscribeRequest) {
    const result = await this.shopAssistantService.transcribe(body.voice_file_url);
    return parseOrThrow(ShopTranscribeResponseSchema, result, 'shop-assistant.transcribe.response');
  }

  @Post('refine-query')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopRefineQueryRequestSchema))
  async refineQuery(@Body() body: ShopRefineQueryRequest) {
    const result = await this.shopAssistantService.refineQuery(
      body.user_text,
      body.previous_params,
      body.role,
      body.prompt_content,
      body.model,
    );
    return parseOrThrow(ShopRefineQueryResponseSchema, result, 'shop-assistant.refine-query.response');
  }

  @Post('search')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopSearchRequestSchema))
  async search(@Body() body: ShopSearchRequest) {
    const result = await this.shopAssistantService.search(body.query_text, body.limit);
    return parseOrThrow(ShopSearchResponseSchema, result, 'shop-assistant.search.response');
  }

  @Post('format-presentation')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopPresentationRequestSchema))
  async formatPresentation(@Body() body: ShopPresentationRequest) {
    const result = await this.shopAssistantService.formatPresentation(
      body.results,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
    );
    return parseOrThrow(ShopPresentationResponseSchema, result, 'shop-assistant.format-presentation.response');
  }

  @Post('compare-prices')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopComparePricesRequestSchema))
  async comparePrices(@Body() body: ShopComparePricesRequest) {
    const result = await this.shopAssistantService.comparePrices(
      body.results,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
      body.priority_order,
    );
    return parseOrThrow(ShopComparePricesResponseSchema, result, 'shop-assistant.compare-prices.response');
  }

  @Post('extract-location')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopExtractLocationRequestSchema))
  async extractLocation(@Body() body: ShopExtractLocationRequest) {
    const result = await this.shopAssistantService.extractLocation(
      body.user_text,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
      body.priority_order,
    );
    return parseOrThrow(ShopExtractLocationResponseSchema, result, 'shop-assistant.extract-location.response');
  }
}
