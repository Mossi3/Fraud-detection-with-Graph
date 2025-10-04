# Personal Tracker Microservices

A Multi-Module Spring Boot Project for Personal Growth Tracker Microservice Architecture.

## Tech Stack

- Java 21
- Spring Boot 3.2+
- Maven
- Spring Cloud Gateway
- Spring AI (OpenAI)
- H2 Database (In-memory)

## Project Structure

```
personal-tracker-ms/
├── pom.xml                 # Root Maven project
├── gateway-service/        # API Gateway (Port: 8080)
├── financial-service/      # Financial tracking (Port: 8081)
├── physical-service/       # Physical health tracking (Port: 8082)
├── emotional-service/      # Emotional wellbeing tracking (Port: 8083)
├── career-service/         # Career development tracking (Port: 8084)
├── spiritual-service/      # Spiritual growth tracking (Port: 8085)
├── social-service/         # Social connections tracking (Port: 8086)
└── ai-service/            # AI-powered insights (Port: 8087)
```

## Getting Started

### Prerequisites

- Java 21
- Maven 3.8+
- OpenAI API Key (for AI service)

### Building the Project

```bash
cd personal-tracker-ms
mvn clean install
```

### Running Individual Services

Each service can be run independently:

```bash
# Gateway Service
cd gateway-service
mvn spring-boot:run

# Financial Service
cd financial-service
mvn spring-boot:run

# And so on for other services...
```

### Running All Services

You can create a script to start all services or use a tool like Docker Compose (configuration not included).

## Service Endpoints

All services are accessible through the Gateway service running on port 8080:

- Financial Service: `http://localhost:8080/api/financial/**`
- Physical Service: `http://localhost:8080/api/physical/**`
- Emotional Service: `http://localhost:8080/api/emotional/**`
- Career Service: `http://localhost:8080/api/career/**`
- Spiritual Service: `http://localhost:8080/api/spiritual/**`
- Social Service: `http://localhost:8080/api/social/**`
- AI Service: `http://localhost:8080/api/ai/**`

## H2 Console Access

Each service (except Gateway and AI) has H2 console enabled for database inspection:

- Financial: `http://localhost:8081/h2-console`
- Physical: `http://localhost:8082/h2-console`
- Emotional: `http://localhost:8083/h2-console`
- Career: `http://localhost:8084/h2-console`
- Spiritual: `http://localhost:8085/h2-console`
- Social: `http://localhost:8086/h2-console`

Default credentials:
- Username: `sa`
- Password: (empty)

## Configuration

### AI Service

The AI service requires an OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY=your-api-key-here
```

Or update the `application.properties` file in the AI service.

## Development

Each service is a standalone Spring Boot application with its own:
- Application class
- Configuration
- Database (H2 in-memory)
- Port assignment

Services communicate through the Gateway service, which routes requests based on URL patterns.

## Next Steps

1. Add REST controllers to each service
2. Define domain models and repositories
3. Implement service-to-service communication
4. Add security (Spring Security/OAuth2)
5. Implement service discovery (Eureka)
6. Add monitoring and logging
7. Create Docker configurations
8. Set up CI/CD pipeline