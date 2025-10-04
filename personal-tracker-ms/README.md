# Personal Growth Tracker Microservices

A multi-module Spring Boot project for tracking personal growth across different life dimensions.

## Project Structure

This is a multi-module Maven project with the following microservices:

### Root Project
- **personal-tracker-ms**: Root Maven project with centralized dependency management

### Microservices

1. **gateway-service** (Port: 8080)
   - Spring Cloud Gateway for routing requests
   - Routes traffic to all other microservices

2. **financial-service** (Port: 8081)
   - Financial tracking and budgeting
   - Spring Web, Spring Data JPA, H2 Database

3. **physical-service** (Port: 8082)
   - Physical health and fitness tracking
   - Spring Web, Spring Data JPA, H2 Database

4. **emotional-service** (Port: 8083)
   - Emotional wellness and mood tracking
   - Spring Web, Spring Data JPA, H2 Database

5. **career-service** (Port: 8084)
   - Career development and goal tracking
   - Spring Web, Spring Data JPA, H2 Database

6. **spiritual-service** (Port: 8085)
   - Spiritual growth and mindfulness tracking
   - Spring Web, Spring Data JPA, H2 Database

7. **social-service** (Port: 8086)
   - Social connections and relationships tracking
   - Spring Web, Spring Data JPA, H2 Database

8. **ai-service** (Port: 8087)
   - AI-powered insights and recommendations
   - Spring Web, Spring AI OpenAI Starter

## Technology Stack

- **Java**: 21
- **Spring Boot**: 3.2.0
- **Spring Cloud**: 2023.0.0
- **Spring AI**: 1.0.0-M3
- **Maven**: Multi-module project
- **H2 Database**: In-memory database for development
- **Spring Cloud Gateway**: API Gateway

## Getting Started

### Prerequisites
- Java 21
- Maven 3.9+

### Building the Project
```bash
mvn clean compile
```

### Running Individual Services
```bash
# Gateway Service
cd gateway-service
mvn spring-boot:run

# Financial Service
cd financial-service
mvn spring-boot:run

# Physical Service
cd physical-service
mvn spring-boot:run

# Emotional Service
cd emotional-service
mvn spring-boot:run

# Career Service
cd career-service
mvn spring-boot:run

# Spiritual Service
cd spiritual-service
mvn spring-boot:run

# Social Service
cd social-service
mvn spring-boot:run

# AI Service
cd ai-service
mvn spring-boot:run
```

### API Endpoints

All services are accessible through the gateway at `http://localhost:8080`:

- Financial: `http://localhost:8080/api/financial/**`
- Physical: `http://localhost:8080/api/physical/**`
- Emotional: `http://localhost:8080/api/emotional/**`
- Career: `http://localhost:8080/api/career/**`
- Spiritual: `http://localhost:8080/api/spiritual/**`
- Social: `http://localhost:8080/api/social/**`
- AI: `http://localhost:8080/api/ai/**`

### Database Access

Each service with a database has H2 console enabled:
- Financial: `http://localhost:8081/h2-console`
- Physical: `http://localhost:8082/h2-console`
- Emotional: `http://localhost:8083/h2-console`
- Career: `http://localhost:8084/h2-console`
- Spiritual: `http://localhost:8085/h2-console`
- Social: `http://localhost:8086/h2-console`

### AI Service Configuration

The AI service requires an OpenAI API key. Set the environment variable:
```bash
export OPENAI_API_KEY=your-api-key-here
```

## Project Architecture

This project follows a microservices architecture pattern with:

- **API Gateway**: Single entry point for all client requests
- **Service Discovery**: Each service runs on a different port
- **Database per Service**: Each service has its own H2 database
- **Centralized Configuration**: All dependencies managed in root pom.xml
- **Independent Deployment**: Each service can be built and deployed independently

## Development Notes

- All services use Java 21 and Spring Boot 3.2+
- H2 databases are configured for development with console access
- Spring AI integration is ready for OpenAI API calls
- Gateway routes are pre-configured for all services
- Each service has its own package structure under `com.personaltracker`