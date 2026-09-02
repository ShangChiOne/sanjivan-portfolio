-- ============================================================
-- Course Recovery System (CRS) - Database Schema
-- MySQL 8.x
-- ============================================================

CREATE DATABASE IF NOT EXISTS crs_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE crs_db;

-- ------------------------------------------------------------
-- 1. System Users (Course Administrator & Academic Officer)
-- ------------------------------------------------------------
CREATE TABLE users (
    user_id     INT AUTO_INCREMENT PRIMARY KEY,
    username    VARCHAR(50)  UNIQUE NOT NULL,
    password    VARCHAR(255) NOT NULL,       -- SHA-256 hex hash
    email       VARCHAR(100) UNIQUE NOT NULL,
    full_name   VARCHAR(100) NOT NULL,
    role        ENUM('COURSE_ADMIN', 'ACADEMIC_OFFICER') NOT NULL,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 2. Students
-- ------------------------------------------------------------
CREATE TABLE students (
    student_id      INT AUTO_INCREMENT PRIMARY KEY,
    student_number  VARCHAR(20)  UNIQUE NOT NULL,
    full_name       VARCHAR(100) NOT NULL,
    email           VARCHAR(100) UNIQUE NOT NULL,
    program         VARCHAR(100) NOT NULL,
    year_of_study   INT NOT NULL DEFAULT 1,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 3. Courses
-- ------------------------------------------------------------
CREATE TABLE courses (
    course_id    INT AUTO_INCREMENT PRIMARY KEY,
    course_code  VARCHAR(20)  UNIQUE NOT NULL,
    course_title VARCHAR(100) NOT NULL,
    credit_hours INT NOT NULL,
    is_active    BOOLEAN DEFAULT TRUE
);

-- ------------------------------------------------------------
-- 4. Assessment Components (parts of a course: assignment, exam…)
-- ------------------------------------------------------------
CREATE TABLE assessment_components (
    component_id   INT AUTO_INCREMENT PRIMARY KEY,
    course_id      INT NOT NULL,
    component_name VARCHAR(100) NOT NULL,
    component_type ENUM('ASSIGNMENT', 'EXAM', 'PROJECT', 'QUIZ') NOT NULL,
    weightage      DECIMAL(5,2) NOT NULL,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 5. Student Course Enrollments (one row per attempt per course)
-- ------------------------------------------------------------
CREATE TABLE student_courses (
    enrollment_id   INT AUTO_INCREMENT PRIMARY KEY,
    student_id      INT NOT NULL,
    course_id       INT NOT NULL,
    semester        INT NOT NULL,
    year_of_study   INT NOT NULL,
    attempt_number  INT NOT NULL DEFAULT 1,  -- 1, 2, or 3
    grade           VARCHAR(5),              -- A, A-, B+, B, …, F
    grade_point     DECIMAL(3,2),            -- 0.00 – 4.00
    is_passed       BOOLEAN DEFAULT FALSE,
    enrollment_date DATE,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id)  REFERENCES courses(course_id),
    UNIQUE KEY uq_student_course_attempt (student_id, course_id, semester, year_of_study, attempt_number)
);

-- ------------------------------------------------------------
-- 6. Component Grades per Enrollment
-- ------------------------------------------------------------
CREATE TABLE student_component_grades (
    grade_id       INT AUTO_INCREMENT PRIMARY KEY,
    enrollment_id  INT NOT NULL,
    component_id   INT NOT NULL,
    marks_obtained DECIMAL(5,2),
    marks_total    DECIMAL(5,2),
    is_passed      BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (enrollment_id) REFERENCES student_courses(enrollment_id) ON DELETE CASCADE,
    FOREIGN KEY (component_id)  REFERENCES assessment_components(component_id)
);

-- ------------------------------------------------------------
-- 7. Course Recovery Plans
-- ------------------------------------------------------------
CREATE TABLE recovery_plans (
    plan_id        INT AUTO_INCREMENT PRIMARY KEY,
    student_id     INT NOT NULL,
    enrollment_id  INT NOT NULL,
    created_by     INT NOT NULL,  -- user_id of Academic Officer
    status         ENUM('ACTIVE', 'COMPLETED', 'FAILED') DEFAULT 'ACTIVE',
    recommendation TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id)    REFERENCES students(student_id),
    FOREIGN KEY (enrollment_id) REFERENCES student_courses(enrollment_id),
    FOREIGN KEY (created_by)    REFERENCES users(user_id)
);

-- ------------------------------------------------------------
-- 8. Recovery Plan Milestones
-- ------------------------------------------------------------
CREATE TABLE recovery_milestones (
    milestone_id     INT AUTO_INCREMENT PRIMARY KEY,
    plan_id          INT NOT NULL,
    study_week       VARCHAR(50)  NOT NULL,  -- e.g. "Week 1-2"
    task_description VARCHAR(255) NOT NULL,
    due_date         DATE,
    is_completed     BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (plan_id) REFERENCES recovery_plans(plan_id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 9. Failed Components Targeted by a Recovery Plan
-- ------------------------------------------------------------
CREATE TABLE recovery_failed_components (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    plan_id         INT NOT NULL,
    component_id    INT NOT NULL,
    recovery_grade  DECIMAL(5,2),
    is_recovered    BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (plan_id)       REFERENCES recovery_plans(plan_id) ON DELETE CASCADE,
    FOREIGN KEY (component_id)  REFERENCES assessment_components(component_id)
);

-- ------------------------------------------------------------
-- 10. Email Notification Log
-- ------------------------------------------------------------
CREATE TABLE email_notifications (
    notification_id   INT AUTO_INCREMENT PRIMARY KEY,
    recipient_email   VARCHAR(100) NOT NULL,
    subject           VARCHAR(255) NOT NULL,
    body              TEXT NOT NULL,
    notification_type ENUM(
        'ACCOUNT_CREATED',
        'PASSWORD_RESET',
        'RECOVERY_PLAN',
        'PERFORMANCE_REPORT',
        'ELIGIBILITY_RESULT'
    ) NOT NULL,
    is_sent           BOOLEAN DEFAULT TRUE,
    error_message     TEXT,
    sent_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Seed Data
-- ============================================================

-- Default admin account  (password: Admin@123  → SHA-256)
INSERT INTO users (username, password, email, full_name, role) VALUES
('admin',   SHA2('Admin@123', 256),   'admin@crs.edu',   'System Administrator', 'COURSE_ADMIN'),
('officer', SHA2('Officer@123', 256), 'officer@crs.edu', 'Academic Officer',      'ACADEMIC_OFFICER');

-- Sample courses
INSERT INTO courses (course_code, course_title, credit_hours) VALUES
('CS201', 'Data Structures',        3),
('CS205', 'Database Systems',       3),
('CS210', 'Software Engineering I', 3),
('MA202', 'Discrete Mathematics',   4),
('EN201', 'Academic Writing',       2);

-- Assessment components for CS201
INSERT INTO assessment_components (course_id, component_name, component_type, weightage) VALUES
(1, 'Assignment 1', 'ASSIGNMENT', 20.00),
(1, 'Midterm Exam', 'EXAM',       30.00),
(1, 'Final Exam',   'EXAM',       50.00);

-- Sample student
INSERT INTO students (student_number, full_name, email, program, year_of_study) VALUES
('2025A1234', 'Alex Tan', 'alex.tan@student.edu', 'Bachelor of Computer Science', 1);
