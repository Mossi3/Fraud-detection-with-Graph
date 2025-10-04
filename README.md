# Personal Growth Tracker Microservices

A comprehensive multi-module Spring Boot microservices project for tracking personal growth across various life dimensions.

## Project Overview

This project implements a microservices architecture for a Personal Growth Tracker application using Spring Boot 3.2+, Java 21, and Maven. The system is designed to help users track and analyze their progress across different aspects of personal development.

## Architecture

### Tech Stack
- **Java**: 21
- **Spring Boot**: 3.2.10
- **Spring Cloud**: 2023.0.4
- **Spring AI**: 1.0.0-M2
- **Build Tool**: Maven
- **Database**: H2 (in-memory for each service)
- **Gateway**: Spring Cloud Gateway

### Microservices

The application consists of 8 microservices, each running on different ports:

| Service | Port | Description | Dependencies |
|---------|------|-------------|--------------|
| **Gateway Service** | 8080 | API Gateway routing requests to microservices | Spring Cloud Gateway, WebFlux |
| **Financial Service** | 8081 | Financial goal and expense tracking | Spring Web, JPA, H2 |
| **Physical Service** | 8082 | Physical health and fitness tracking | Spring Web, JPA, H2 |
| **Emotional Service** | 8083 | Emotional well-being monitoring | Spring Web, JPA, H2 |
| **Career Service** | 8084 | Career development and goal tracking | Spring Web, JPA, H2 |
| **Spiritual Service** | 8085 | Spiritual growth and practices | Spring Web, JPA, H2 |
| **Social Service** | 8086 | Social interactions and relationships | Spring Web, JPA, H2 |
| **AI Service** | 8087 | AI-powered insights and recommendations | Spring Web, Spring AI OpenAI |

## Project Structure

```
personal-tracker-ms/
├── pom.xml                     # Root POM with dependency management
├── gateway-service/            # API Gateway
│   ├── pom.xml
│   └── src/
├── financial-service/          # Financial tracking microservice
│   ├── pom.xml
│   └── src/
├── physical-service/           # Physical health microservice
│   ├── pom.xml
│   └── src/
├── emotional-service/          # Emotional well-being microservice
│   ├── pom.xml
│   └── src/
├── career-service/             # Career development microservice
│   ├── pom.xml
│   └── src/
├── spiritual-service/          # Spiritual growth microservice
│   ├── pom.xml
│   └── src/
├── social-service/             # Social interactions microservice
│   ├── pom.xml
│   └── src/
└── ai-service/                 # AI insights microservice
    ├── pom.xml
    └── src/
```

## Gateway Routing Configuration

The Gateway Service routes requests to appropriate microservices based on URL patterns:

| Path Pattern | Target Service | Port |
|--------------|----------------|------|
| `/api/financial/**` | Financial Service | 8081 |
| `/api/physical/**` | Physical Service | 8082 |
| `/api/emotional/**` | Emotional Service | 8083 |
| `/api/career/**` | Career Service | 8084 |
| `/api/spiritual/**` | Spiritual Service | 8085 |
| `/api/social/**` | Social Service | 8086 |
| `/api/ai/**` | AI Service | 8087 |

## Getting Started

### Prerequisites
- Java 21
- Maven 3.6+
- (Optional) OpenAI API Key for AI Service functionality

### Building the Project

1. **Validate Project Structure**:
   ```bash
   mvn validate
   ```

2. **Clean and Compile All Services**:
   ```bash
   mvn clean compile
   ```

3. **Package All Services**:
   ```bash
   mvn clean package
   ```

### Running the Services

#### Start All Services (Recommended Order)

1. **Start Gateway Service**:
   ```bash
   cd gateway-service
   mvn spring-boot:run
   ```

2. **Start Individual Microservices** (in separate terminals):
   ```bash
   # Financial Service
   cd financial-service && mvn spring-boot:run

   # Physical Service  
   cd physical-service && mvn spring-boot:run

   # Emotional Service
   cd emotional-service && mvn spring-boot:run

   # Career Service
   cd career-service && mvn spring-boot:run

   # Spiritual Service
   cd spiritual-service && mvn spring-boot:run

   # Social Service
   cd social-service && mvn spring-boot:run

   # AI Service (requires OPENAI_API_KEY environment variable)
   cd ai-service && mvn spring-boot:run
   ```

#### Alternative: Run from Root Directory
```bash
# Build all modules
mvn clean install

# Run individual services using their JAR files
java -jar gateway-service/target/gateway-service-1.0.0.jar
java -jar financial-service/target/financial-service-1.0.0.jar
# ... etc for other services
```

### Configuration

#### AI Service Setup
The AI Service requires an OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY=your-openai-api-key-here
```

Or modify the `ai-service/src/main/resources/application.yml` file:
```yaml
spring:
  ai:
    openai:
      api-key: your-actual-api-key-here
```

#### Database Access
Each service uses H2 in-memory database with console access:
- **Financial Service**: http://localhost:8081/h2-console
- **Physical Service**: http://localhost:8082/h2-console  
- **Emotional Service**: http://localhost:8083/h2-console
- **Career Service**: http://localhost:8084/h2-console
- **Spiritual Service**: http://localhost:8085/h2-console
- **Social Service**: http://localhost:8086/h2-console

**Connection Details**:
- JDBC URL: `jdbc:h2:mem:[servicename]db` (e.g., `jdbc:h2:mem:financialdb`)
- Username: `sa`
- Password: `password`

## API Access

### Through Gateway (Recommended)
Access all services through the gateway at `http://localhost:8080`:
- Financial API: `http://localhost:8080/api/financial/...`
- Physical API: `http://localhost:8080/api/physical/...`
- Emotional API: `http://localhost:8080/api/emotional/...`
- Career API: `http://localhost:8080/api/career/...`
- Spiritual API: `http://localhost:8080/api/spiritual/...`
- Social API: `http://localhost:8080/api/social/...`
- AI API: `http://localhost:8080/api/ai/...`

### Direct Service Access
Each service can also be accessed directly on their individual ports (8081-8087).

## Health Checks

Monitor service health through actuator endpoints:
- Gateway: `http://localhost:8080/actuator/health`
- Individual Services: `http://localhost:808[1-7]/actuator/health`

## Development Features

### Centralized Dependency Management
- All Spring Boot and Spring Cloud versions managed in root `pom.xml`
- Consistent dependency versions across all microservices
- Easy version updates from single location

### Development Tools Included
- Spring Boot DevTools (for hot reloading during development)
- Actuator endpoints for monitoring
- H2 console for database inspection
- Debug logging configurations

### Modular Architecture
- Each service is independent with its own database
- Services can be developed, tested, and deployed separately  
- Clean separation of concerns by business domain

## Next Steps for Development

1. **Add Domain Models**: Create JPA entities for each service domain
2. **Implement REST Controllers**: Add API endpoints for CRUD operations
3. **Service Communication**: Implement inter-service communication using OpenFeign
4. **Security**: Add Spring Security and JWT authentication
5. **Monitoring**: Integrate distributed tracing with Zipkin/Sleuth
6. **Configuration**: Add Spring Cloud Config Server
7. **Service Discovery**: Implement Eureka or Consul for service registry
8. **Testing**: Add comprehensive unit and integration tests
9. **Documentation**: Generate API documentation with OpenAPI/Swagger
10. **Deployment**: Create Docker containers and Kubernetes manifests

## Contributing

This project follows Spring Boot and microservices best practices. Each service should remain focused on its specific domain and communicate through well-defined APIs.

---

**Created**: October 2025  
**Java Version**: 21  
**Spring Boot Version**: 3.2.10  
**Spring Cloud Version**: 2023.0.4