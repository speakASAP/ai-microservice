import { JwtUtil } from '../../src/service-identity/jwt.util';

const SECRET = 'test-secret-at-least-32-chars-long!!';
const SERVICE_ID = 'runlayer';

describe('JwtUtil', () => {
  describe('sign', () => {
    it('produces a 3-part JWT', () => {
      const token = JwtUtil.sign(SERVICE_ID, SECRET);
      expect(token.split('.')).toHaveLength(3);
    });

    it('round-trips serviceId through verify', () => {
      const token = JwtUtil.sign(SERVICE_ID, SECRET);
      const payload = JwtUtil.verify(token, SECRET);
      expect(payload.serviceId).toBe(SERVICE_ID);
      expect(payload.iss).toBe('ai-microservice');
    });
  });

  describe('verify', () => {
    it('throws on malformed token', () => {
      expect(() => JwtUtil.verify('not.a.valid.jwt.here', SECRET)).toThrow('Malformed token');
    });

    it('throws on wrong secret', () => {
      const token = JwtUtil.sign(SERVICE_ID, SECRET);
      expect(() => JwtUtil.verify(token, 'wrong-secret')).toThrow('Invalid signature');
    });

    it('throws on expired token', () => {
      // Sign with -1s expiry (already expired)
      const token = JwtUtil.sign(SERVICE_ID, SECRET, -1);
      expect(() => JwtUtil.verify(token, SECRET)).toThrow('Token expired');
    });

    it('throws on tampered payload', () => {
      const token = JwtUtil.sign(SERVICE_ID, SECRET);
      const parts = token.split('.');
      // Replace payload with a different serviceId
      const tampered = Buffer.from(
        JSON.stringify({ serviceId: 'attacker', iss: 'ai-microservice', iat: 0, exp: 9999999999 }),
      ).toString('base64url');
      const tamperedToken = `${parts[0]}.${tampered}.${parts[2]}`;
      expect(() => JwtUtil.verify(tamperedToken, SECRET)).toThrow('Invalid signature');
    });
  });
});
