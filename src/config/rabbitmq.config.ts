import { RabbitMQConfig } from '@golevelup/nestjs-rabbitmq';

export function getRabbitMQConfig(): RabbitMQConfig {
  return {
    uri: process.env.RABBITMQ_URL || 'amqp://guest:guest@localhost:5672',
    exchanges: [
      {
        name: 'claude-code-exchange',
        type: 'direct',
        options: { durable: true },
      },
    ],
    queues: [
      {
        name: 'claude-code-execute-queue',
        options: {
          durable: true,
          arguments: {
            'x-dead-letter-exchange': 'claude-code-dlx',
          },
        },
        exchange: 'claude-code-exchange',
        routingKey: 'claude-code.execute',
      },
    ],
    connectionInitOptions: { wait: false },
  };
}
