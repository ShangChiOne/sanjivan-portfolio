package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "students")
@NamedQueries({
    @NamedQuery(name = "Student.findAll",
                query = "SELECT s FROM Student s WHERE s.isActive = true ORDER BY s.fullName"),
    @NamedQuery(name = "Student.findByNumber",
                query = "SELECT s FROM Student s WHERE s.studentNumber = :studentNumber"),
    @NamedQuery(name = "Student.findByEmail",
                query = "SELECT s FROM Student s WHERE s.email = :email")
})
public class Student implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "student_id")
    private Integer studentId;

    @Column(name = "student_number", nullable = false, unique = true, length = 20)
    private String studentNumber;

    @Column(name = "full_name", nullable = false, length = 100)
    private String fullName;

    @Column(name = "email", nullable = false, unique = true, length = 100)
    private String email;

    @Column(name = "program", nullable = false, length = 100)
    private String program;

    @Column(name = "year_of_study", nullable = false)
    private int yearOfStudy;

    @Column(name = "is_active")
    private boolean isActive = true;

    @Column(name = "status", length = 20)
    private String status = "ACTIVE";

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @OneToMany(mappedBy = "student", fetch = FetchType.LAZY)
    private List<StudentCourse> enrollments;

    @PrePersist
    protected void onCreate() { createdAt = LocalDateTime.now(); }

    // ---- Getters & Setters ----

    public Integer getStudentId()                       { return studentId; }
    public void setStudentId(Integer studentId)         { this.studentId = studentId; }
    public String getStudentNumber()                    { return studentNumber; }
    public void setStudentNumber(String studentNumber)  { this.studentNumber = studentNumber; }
    public String getFullName()                         { return fullName; }
    public void setFullName(String fullName)            { this.fullName = fullName; }
    public String getEmail()                            { return email; }
    public void setEmail(String email)                  { this.email = email; }
    public String getProgram()                          { return program; }
    public void setProgram(String program)              { this.program = program; }
    public int getYearOfStudy()                         { return yearOfStudy; }
    public void setYearOfStudy(int yearOfStudy)         { this.yearOfStudy = yearOfStudy; }
    public boolean isActive()                           { return isActive; }
    public void setActive(boolean active)               { isActive = active; }
    public String getStatus()                           { return status != null ? status : "ACTIVE"; }
    public void setStatus(String status)                { this.status = status; }
    public LocalDateTime getCreatedAt()                 { return createdAt; }
    public List<StudentCourse> getEnrollments()         { return enrollments; }
    public void setEnrollments(List<StudentCourse> e)   { this.enrollments = e; }
}
