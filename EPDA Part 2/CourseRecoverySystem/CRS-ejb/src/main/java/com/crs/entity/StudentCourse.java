package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;
import java.time.LocalDate;
import java.util.List;

/**
 * Represents one enrollment attempt of a student in a course.
 * attempt_number: 1 (normal), 2 (failed component only), 3 (full resit).
 */
@Entity
@Table(name = "student_courses",
       uniqueConstraints = @UniqueConstraint(
               columnNames = {"student_id", "course_id", "semester", "year_of_study", "attempt_number"}))
@NamedQueries({
    @NamedQuery(name = "StudentCourse.findByStudent",
                query = "SELECT sc FROM StudentCourse sc WHERE sc.student.studentId = :studentId ORDER BY sc.yearOfStudy, sc.semester"),
    @NamedQuery(name = "StudentCourse.findFailedByStudent",
                query = "SELECT sc FROM StudentCourse sc WHERE sc.student.studentId = :studentId AND sc.isPassed = false"),
    @NamedQuery(name = "StudentCourse.countFailedByStudent",
                query = "SELECT COUNT(sc) FROM StudentCourse sc WHERE sc.student.studentId = :studentId AND sc.isPassed = false AND sc.attemptNumber = (SELECT MAX(sc2.attemptNumber) FROM StudentCourse sc2 WHERE sc2.student.studentId = :studentId AND sc2.course.courseId = sc.course.courseId)")
})
public class StudentCourse implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "enrollment_id")
    private Integer enrollmentId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "student_id", nullable = false)
    private Student student;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "course_id", nullable = false)
    private Course course;

    @Column(name = "semester", nullable = false)
    private int semester;

    @Column(name = "year_of_study", nullable = false)
    private int yearOfStudy;

    @Column(name = "attempt_number", nullable = false)
    private int attemptNumber = 1;

    @Column(name = "grade", length = 5)
    private String grade;

    @Column(name = "grade_point", precision = 3, scale = 2)
    private Double gradePoint;

    @Column(name = "is_passed")
    private boolean isPassed = false;

    @Column(name = "enrollment_date")
    private LocalDate enrollmentDate;

    @OneToMany(mappedBy = "enrollment", fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    private List<StudentComponentGrade> componentGrades;

    @OneToMany(mappedBy = "enrollment", fetch = FetchType.LAZY)
    private List<RecoveryPlan> recoveryPlans;

    // ---- Getters & Setters ----

    public Integer getEnrollmentId()                        { return enrollmentId; }
    public void setEnrollmentId(Integer id)                 { this.enrollmentId = id; }
    public Student getStudent()                             { return student; }
    public void setStudent(Student student)                 { this.student = student; }
    public Course getCourse()                               { return course; }
    public void setCourse(Course course)                    { this.course = course; }
    public int getSemester()                                { return semester; }
    public void setSemester(int semester)                   { this.semester = semester; }
    public int getYearOfStudy()                             { return yearOfStudy; }
    public void setYearOfStudy(int yearOfStudy)             { this.yearOfStudy = yearOfStudy; }
    public int getAttemptNumber()                           { return attemptNumber; }
    public void setAttemptNumber(int attemptNumber)         { this.attemptNumber = attemptNumber; }
    public String getGrade()                                { return grade; }
    public void setGrade(String grade)                      { this.grade = grade; }
    public Double getGradePoint()                           { return gradePoint; }
    public void setGradePoint(Double gradePoint)            { this.gradePoint = gradePoint; }
    public boolean isPassed()                               { return isPassed; }
    public void setPassed(boolean passed)                   { isPassed = passed; }
    public LocalDate getEnrollmentDate()                    { return enrollmentDate; }
    public void setEnrollmentDate(LocalDate d)              { this.enrollmentDate = d; }
    public List<StudentComponentGrade> getComponentGrades() { return componentGrades; }
    public void setComponentGrades(List<StudentComponentGrade> g) { this.componentGrades = g; }
    public List<RecoveryPlan> getRecoveryPlans()            { return recoveryPlans; }
    public void setRecoveryPlans(List<RecoveryPlan> r)      { this.recoveryPlans = r; }
}
