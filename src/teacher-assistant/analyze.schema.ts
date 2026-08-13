export const ANALYZE_OUTPUT_SCHEMA = {
  type: 'object',
  required: ['clusters'],
  properties: {
    clusters: {
      type: 'array',
      items: {
        type: 'object',
        required: ['topicSlug', 'title', 'explanation', 'rules', 'examples', 'answers'],
        properties: {
          topicSlug: { type: 'string' },
          title: { type: 'string' },
          explanation: { type: 'string' },
          rules: { type: 'array', items: { type: 'string' } },
          examples: {
            type: 'array',
            items: {
              type: 'object',
              required: ['text', 'gloss'],
              properties: {
                text: { type: 'string' },
                gloss: { type: 'string' },
              },
            },
          },
          answers: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
} as const;
