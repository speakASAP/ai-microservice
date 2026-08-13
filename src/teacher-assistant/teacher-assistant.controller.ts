import { Body, Controller, HttpCode, Post, UseGuards } from '@nestjs/common';
import { ServiceAuthGuard } from '../service-identity/service-auth.guard';
import { GenerateService } from './generate.service';
import { ValidateService } from './validate.service';
import { AnalyzeService } from './analyze.service';
import { GenerateDrillRequestDto } from './dto/generate-drill-request.dto';
import { ValidateDrillRequestDto } from './dto/validate-drill-request.dto';
import { AnalyzeErrorsRequestDto } from './dto/analyze-errors-request.dto';
import { AnalyzeErrorsResponse, GenerateDrillResponse, ValidateDrillResponse } from './contracts';

/**
 * Service-to-service endpoints called by education-service's drill
 * orchestration (Track D) — not browser/admin traffic, hence
 * ServiceAuthGuard (service JWT) rather than AdminAuthGuard.
 */
// Deliberately explicit even though ServiceAuthGuard also runs globally via
// APP_GUARD (see ServiceIdentityModule) — do not remove this as "redundant".
// It documents the guarantee at the call site and is what
// teacher-assistant.controller.spec.ts's guard test actually asserts on.
@UseGuards(ServiceAuthGuard)
@Controller('api/teacher-assistant')
export class TeacherAssistantController {
  constructor(
    private readonly generateService: GenerateService,
    private readonly validateService: ValidateService,
    private readonly analyzeService: AnalyzeService,
  ) {}

  @Post('generate-drill')
  @HttpCode(200)
  generateDrill(@Body() dto: GenerateDrillRequestDto): Promise<GenerateDrillResponse> {
    return this.generateService.generate(dto);
  }

  @Post('validate-drill')
  @HttpCode(200)
  validateDrill(@Body() dto: ValidateDrillRequestDto): Promise<ValidateDrillResponse> {
    return this.validateService.validate(dto);
  }

  @Post('analyze-drill-errors')
  @HttpCode(200)
  analyzeDrillErrors(@Body() dto: AnalyzeErrorsRequestDto): Promise<AnalyzeErrorsResponse> {
    return this.analyzeService.analyze(dto);
  }
}
