import { Body, Controller, HttpCode, Post, UseGuards } from '@nestjs/common';
import { ServiceAuthGuard } from '../service-identity/service-auth.guard';
import { GenerateService } from './generate.service';
import { ValidateService } from './validate.service';
import { GenerateDrillRequestDto } from './dto/generate-drill-request.dto';
import { ValidateDrillRequestDto } from './dto/validate-drill-request.dto';
import { GenerateDrillResponse, ValidateDrillResponse } from './contracts';

/**
 * Service-to-service endpoints called by education-service's drill
 * orchestration (Track D) — not browser/admin traffic, hence
 * ServiceAuthGuard (service JWT) rather than AdminAuthGuard.
 */
@UseGuards(ServiceAuthGuard)
@Controller('api/teacher-assistant')
export class TeacherAssistantController {
  constructor(
    private readonly generateService: GenerateService,
    private readonly validateService: ValidateService,
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
}
