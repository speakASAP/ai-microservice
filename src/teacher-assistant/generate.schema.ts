export const GENERATE_OUTPUT_SCHEMA = {
  type: 'object',
  required: ['items'],
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        required: ['template', 'blanks', 'topicSlug', 'newWords'],
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
          topicSlug: { type: 'string' },
          newWords: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
} as const;
