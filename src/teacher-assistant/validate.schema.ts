export const VALIDATE_OUTPUT_SCHEMA = {
  type: 'object',
  required: ['results'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        required: ['itemRef', 'verdicts', 'issues', 'suggestedFix'],
        properties: {
          itemRef: { type: 'integer' },
          verdicts: {
            type: 'object',
            required: ['topicAlignment', 'grammar', 'level', 'naturalness'],
            properties: {
              topicAlignment: { type: 'string', enum: ['PASS', 'WARN', 'FAIL'] },
              grammar: { type: 'string', enum: ['PASS', 'FAIL'] },
              level: { type: 'string', enum: ['PASS', 'WARN', 'FAIL'] },
              naturalness: { type: 'string', enum: ['PASS', 'WARN', 'FAIL'] },
            },
          },
          issues: {
            type: 'array',
            items: {
              type: 'object',
              required: ['code', 'message'],
              properties: {
                code: {
                  type: 'string',
                  enum: ['OFF_TOPIC', 'UNGRAMMATICAL', 'WRONG_LEVEL', 'UNNATURAL'],
                },
                message: { type: 'string' },
                span: { type: 'string' },
              },
            },
          },
          // A fix without a usable `template` is not a fix — ValidateService
          // treats one as absent and downgrades the item to WARN. Ask for the
          // whole corrected item or for null; never a half-filled object.
          suggestedFix: {
            type: ['object', 'null'],
            required: ['template', 'blanks', 'hint'],
            properties: {
              template: { type: 'string', minLength: 1 },
              blanks: {
                type: 'array',
                items: {
                  type: 'object',
                  required: ['prompt', 'answer', 'alternatives'],
                  properties: {
                    prompt: { type: 'string' },
                    answer: { type: 'string' },
                    alternatives: { type: 'array', items: { type: 'string' } },
                  },
                },
              },
              hint: { type: ['string', 'null'] },
            },
          },
        },
      },
    },
  },
} as const;
