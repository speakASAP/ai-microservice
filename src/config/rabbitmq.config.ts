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
    connectionInitOptions: { wait: false },
  };
}
