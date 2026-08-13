import 'reflect-metadata';
import { ArgumentMetadata, BadRequestException, ValidationPipe } from '@nestjs/common';
import { GUARDS_METADATA } from '@nestjs/common/constants';
import { TeacherAssistantController } from './teacher-assistant.controller';
import { ServiceAuthGuard } from '../service-identity/service-auth.guard';
import { GenerateDrillRequestDto } from './dto/generate-drill-request.dto';
import { ValidateDrillRequestDto } from './dto/validate-drill-request.dto';
import { AnalyzeErrorsRequestDto } from './dto/analyze-errors-request.dto';
import { AnalyzeErrorsRequest, GenerateDrillRequest, ValidateDrillRequest } from './contracts';

// Mirrors the global pipe installed in main.ts — see bootstrap()'s
// `app.useGlobalPipes(new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true, transform: true }))`.
const pipe = new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true, transform: true });

const validGenerateBody: GenerateDrillRequest = {
  languageCode: 'de',
  materialLanguage: 'ru',
  level: 'A2',
  topics: [{ slug: 'prepositions', title: 'Предлоги', focus: 'an, bei, für' }],
  instructions: '10 sentences, present tense only',
  count: 10,
  knownVocabulary: ['bus', 'schule'],
  maxNewWordsPerSentence: 2,
  exampleItems: ['Ich gehe [in]{in} die Schule.'],
  avoidTexts: [],
  correlationId: 'c-1',
};

const validValidateBody: ValidateDrillRequest = {
  languageCode: 'de',
  materialLanguage: 'ru',
  level: 'A2',
  topics: [{ slug: 'prepositions', title: 'Предлоги' }],
  instructions: 'prepositions only',
  items: [
    {
      itemRef: 0,
      template: 'Ich warte [на]{auf} den Bus.',
      blanks: [{ index: 0, prompt: 'на', answer: 'auf', alternatives: [] }],
      hint: null,
    },
  ],
  correlationId: 'c-1',
};

const validAnalyzeBody: AnalyzeErrorsRequest = {
  languageCode: 'en',
  materialLanguage: 'ru',
  level: 'A2',
  allowedTopicSlugs: ['en.prepositions-of-movement', 'en.other'],
  failures: [
    {
      answer: 'through',
      sentence: 'We will have to walk {{0}} this market.',
      prompt: 'через',
      wrongAttempts: ['across'],
      revealed: false,
      mistakeCount: 1,
    },
  ],
  correlationId: 'cid-1',
};

const generateMetadata: ArgumentMetadata = { type: 'body', metatype: GenerateDrillRequestDto, data: '' };
const validateMetadata: ArgumentMetadata = { type: 'body', metatype: ValidateDrillRequestDto, data: '' };
const analyzeMetadata: ArgumentMetadata = { type: 'body', metatype: AnalyzeErrorsRequestDto, data: '' };

describe('TeacherAssistantController guarding', () => {
  it('applies ServiceAuthGuard to the controller', () => {
    // Reflection metadata, not an HTTP round-trip: @UseGuards() stamps
    // GUARDS_METADATA ('__guards__') on the class with the guard classes
    // passed to it. This is what NestJS itself reads at request time to
    // decide which guards run, so asserting on it here is a direct check
    // of "will this route actually be guarded" without needing to boot an
    // HTTP server.
    const guards = Reflect.getMetadata(GUARDS_METADATA, TeacherAssistantController) as unknown[] | undefined;
    expect(guards).toBeDefined();
    expect(guards).toContain(ServiceAuthGuard);
  });
});

describe('TeacherAssistantController request validation (via the global ValidationPipe)', () => {
  it('accepts a well-formed generate-drill body', async () => {
    await expect(pipe.transform({ ...validGenerateBody }, generateMetadata)).resolves.toBeDefined();
  });

  // Every key of GenerateDrillRequest (11 fields) — none of them are `?:` in
  // the contract, so a missing key must always reject.
  const generateRequiredFields: (keyof GenerateDrillRequest)[] = [
    'languageCode',
    'materialLanguage',
    'level',
    'topics',
    'instructions',
    'count',
    'knownVocabulary',
    'maxNewWordsPerSentence',
    'exampleItems',
    'avoidTexts',
    'correlationId',
  ];

  it.each(generateRequiredFields)(
    'rejects a generate-drill body missing %s with 400',
    async (field) => {
      const body: Record<string, unknown> = { ...validGenerateBody };
      delete body[field];
      await expect(pipe.transform(body, generateMetadata)).rejects.toBeInstanceOf(BadRequestException);
    },
  );

  it('accepts a well-formed validate-drill body', async () => {
    await expect(pipe.transform({ ...validValidateBody }, validateMetadata)).resolves.toBeDefined();
  });

  // Every key of ValidateDrillRequest (7 fields) — none of them are `?:` in
  // the contract, so a missing key must always reject.
  const validateRequiredFields: (keyof ValidateDrillRequest)[] = [
    'languageCode',
    'materialLanguage',
    'level',
    'topics',
    'instructions',
    'items',
    'correlationId',
  ];

  it.each(validateRequiredFields)(
    'rejects a validate-drill body missing %s with 400',
    async (field) => {
      const body: Record<string, unknown> = { ...validValidateBody };
      delete body[field];
      await expect(pipe.transform(body, validateMetadata)).rejects.toBeInstanceOf(BadRequestException);
    },
  );

  // Regression coverage for the nested `items[].hint` field, which is
  // `string | null` in the contract — a required key, nullable value. This
  // must reject a missing key while still accepting an explicit null.
  it('rejects a validate-drill item whose hint key is missing entirely', async () => {
    const body = JSON.parse(JSON.stringify(validValidateBody));
    delete body.items[0].hint;
    await expect(pipe.transform(body, validateMetadata)).rejects.toBeInstanceOf(BadRequestException);
  });

  it('accepts a validate-drill item whose hint is explicitly null', async () => {
    const body = JSON.parse(JSON.stringify(validValidateBody));
    body.items[0].hint = null;
    await expect(pipe.transform(body, validateMetadata)).resolves.toBeDefined();
  });

  it('accepts a validate-drill item whose hint is a non-null string', async () => {
    const body = JSON.parse(JSON.stringify(validValidateBody));
    body.items[0].hint = '(warten auf – ждать)';
    await expect(pipe.transform(body, validateMetadata)).resolves.toBeDefined();
  });

  it('accepts a well-formed analyze-drill-errors body', async () => {
    await expect(pipe.transform({ ...validAnalyzeBody }, analyzeMetadata)).resolves.toBeDefined();
  });

  // Every key of AnalyzeErrorsRequest (6 fields) — none of them are `?:` in
  // the contract, so a missing key must always reject.
  const analyzeRequiredFields: (keyof AnalyzeErrorsRequest)[] = [
    'languageCode',
    'materialLanguage',
    'level',
    'allowedTopicSlugs',
    'failures',
    'correlationId',
  ];

  it.each(analyzeRequiredFields)(
    'rejects an analyze-drill-errors body missing %s with 400',
    async (field) => {
      const body: Record<string, unknown> = { ...validAnalyzeBody };
      delete body[field];
      await expect(pipe.transform(body, analyzeMetadata)).rejects.toBeInstanceOf(BadRequestException);
    },
  );

  it('rejects an analyze-drill-errors body with an empty allowedTopicSlugs', async () => {
    const body = { ...validAnalyzeBody, allowedTopicSlugs: [] };
    await expect(pipe.transform(body, analyzeMetadata)).rejects.toBeInstanceOf(BadRequestException);
  });

  it('rejects an analyze-drill-errors body with an empty failures list', async () => {
    const body = { ...validAnalyzeBody, failures: [] };
    await expect(pipe.transform(body, analyzeMetadata)).rejects.toBeInstanceOf(BadRequestException);
  });
});

describe('TeacherAssistantController delegation', () => {
  it('delegates generate-drill to GenerateService.generate exactly once', async () => {
    const generateService = { generate: jest.fn().mockResolvedValue({ items: [], meta: {} }) } as any;
    const validateService = { validate: jest.fn() } as any;
    const analyzeService = { analyze: jest.fn() } as any;
    const controller = new TeacherAssistantController(generateService, validateService, analyzeService);

    const dto = Object.assign(new GenerateDrillRequestDto(), validGenerateBody);
    const result = await controller.generateDrill(dto);

    expect(generateService.generate).toHaveBeenCalledTimes(1);
    expect(generateService.generate).toHaveBeenCalledWith(dto);
    expect(validateService.validate).not.toHaveBeenCalled();
    expect(analyzeService.analyze).not.toHaveBeenCalled();
    expect(result).toEqual({ items: [], meta: {} });
  });

  it('delegates validate-drill to ValidateService.validate exactly once', async () => {
    const generateService = { generate: jest.fn() } as any;
    const validateService = { validate: jest.fn().mockResolvedValue({ results: [], meta: {} }) } as any;
    const analyzeService = { analyze: jest.fn() } as any;
    const controller = new TeacherAssistantController(generateService, validateService, analyzeService);

    const dto = Object.assign(new ValidateDrillRequestDto(), validValidateBody);
    const result = await controller.validateDrill(dto);

    expect(validateService.validate).toHaveBeenCalledTimes(1);
    expect(validateService.validate).toHaveBeenCalledWith(dto);
    expect(generateService.generate).not.toHaveBeenCalled();
    expect(analyzeService.analyze).not.toHaveBeenCalled();
    expect(result).toEqual({ results: [], meta: {} });
  });

  it('routes analyze-drill-errors to the analyze service', async () => {
    const generateService = { generate: jest.fn() } as any;
    const validateService = { validate: jest.fn() } as any;
    const analyzeService = { analyze: jest.fn().mockResolvedValue({ clusters: [], meta: {} }) } as any;
    const controller = new TeacherAssistantController(generateService, validateService, analyzeService);

    const dto = Object.assign(new AnalyzeErrorsRequestDto(), validAnalyzeBody);
    const result = await controller.analyzeDrillErrors(dto);

    expect(analyzeService.analyze).toHaveBeenCalledTimes(1);
    expect(analyzeService.analyze).toHaveBeenCalledWith(dto);
    expect(generateService.generate).not.toHaveBeenCalled();
    expect(validateService.validate).not.toHaveBeenCalled();
    expect(result).toEqual({ clusters: [], meta: {} });
  });
});
