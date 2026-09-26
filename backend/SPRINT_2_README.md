# Sprint 2 — Authentication & Security

## 1. Sprint Overview

Sprint 2 focuses on implementing the authentication and security foundation of the OptiWealth backend.

The main objective of this sprint was to provide secure user authentication using password hashing and JWT-based authentication.

---

## 2. Sprint Goal

The goal of Sprint 2 was to implement:

- Secure password hashing
- User registration
- User login
- JWT access-token generation
- JWT token verification
- Bearer authentication
- Protected user endpoints
- Current authenticated-user retrieval
- Email validation
- Authentication test cases

---

## 3. Scope of This Sprint

### Backend Team

This sprint covers the backend authentication and security layer.

### ML Team

Machine learning, risk scoring, portfolio optimization, model training, and related ML functionality are handled separately by the ML team and are not part of this sprint.

---

# 4. Features Implemented

## 4.1 Password Hashing

Passwords are never stored directly in the database.

The backend uses `pwdlib` with the Argon2 password hashing algorithm.

Password flow:

```text
User Password
      ↓
Argon2 Hashing
      ↓
password_hash
      ↓
PostgreSQL
