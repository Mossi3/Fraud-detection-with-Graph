# Personal Growth Tracker Microservices

A multi-module Spring Boot project for tracking personal growth across multiple dimensions using microservice architecture.

## Project Structure

This is a Maven multi-module project with the following services:

### Core Services
- **gateway-service** - Spring Cloud Gateway for routing requests to microservices
- **financial-service** - Tracks financial goals and progress
- **physical-service** - Tracks physical fitness and health goals
- **emotional-service** - Tracks emotional wellbeing and mental health
- **career-service** - Tracks career development and professional goals
- **spiritual-service** - Tracks spiritual growth and mindfulness
- **social-service** - Tracks social connections and relationships
- **ai-service** - AI-powered insights and recommendations using OpenAI

## Technology Stack

- **Java 21**
- **Spring Boot 3.2+**
- **Spring Cloud Gateway** (for API gateway)
- **Spring Data JPA** (for data persistence)
- **H2 Database** (in-memory database for development)
- **Spring AI** (for AI-powered features)
- **Maven** (for dependency management)

## Prerequisites

- Java 21 or higher
- Maven 3.6+
- OpenAI API key (for AI service)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd personal-tracker-ms
   ```

2. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY=your-openai-api-key-here
   ```

3. **Build the project**
   ```bash
   mvn clean compile
   ```

4. **Run the services**

   Start each service in a separate terminal:

   ```bash
   # Gateway Service (port 8080)
   mvn spring-boot:run -pl gateway-service

   # Financial Service (port 8081)
   mvn spring-boot:run -pl financial-service

   # Physical Service (port 8082)
   mvn spring-boot:run -pl physical-service

   # Emotional Service (port 8083)
   mvn spring-boot:run -pl emotional-service

   # Career Service (port 8084)
   mvn spring-boot:run -pl career-service

   # Spiritual Service (port 8085)
   mvn spring-boot:run -pl spiritual-service

   # Social Service (port 8086)
   mvn spring-boot:run -pl social-service

   # AI Service (port 8087)
   mvn spring-boot:run -pl ai-service
   ```

## API Gateway Routes

The gateway service routes requests as follows:

- `http://localhost:8080/api/financial/**` → Financial Service
- `http://localhost:8080/api/physical/**` → Physical Service
- `http://localhost:8080/api/emotional/**` → Emotional Service
- `http://localhost:8080/api/career/**` → Career Service
- `http://localhost:8080/api/spiritual/**` → Spiritual Service
- `http://localhost:8080/api/social/**` → Social Service
- `http://localhost:8080/api/ai/**` → AI Service

## Database Access

Each service uses H2 in-memory database. Access the H2 console at:
- Financial Service: `http://localhost:8081/h2-console`
- Physical Service: `http://localhost:8082/h2-console`
- (Other services follow the same pattern)

Default credentials:
- JDBC URL: `jdbc:h2:mem:{servicename}db`
- Username: `sa`
- Password: (empty)

## Development

### Adding New Endpoints

Each service follows standard Spring Boot patterns. Add new controllers, services, and repositories as needed.

### Database Schema

JPA entities will be automatically created based on your entity classes. The schema is generated at runtime.

## Contributing

1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Submit a pull request

## License

This project is licensed under the MIT License.