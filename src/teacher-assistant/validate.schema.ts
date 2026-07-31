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
          suggestedFix: {
            type: ['object', 'null'],
            properties: {
              template: { type: 'string' },
              blanks: {
                type: 'array',
                items: {
                  type: 'object',
                  required: ['prompt', 'answer'],
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
