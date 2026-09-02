package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;

@Entity
@Table(name = "student_component_grades")
@NamedQuery(name = "StudentComponentGrade.findByEnrollment",
            query = "SELECT g FROM StudentComponentGrade g WHERE g.enrollment.enrollmentId = :enrollmentId")
public class StudentComponentGrade implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "grade_id")
    private Integer gradeId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "enrollment_id", nullable = false)
    private StudentCourse enrollment;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "component_id", nullable = false)
    private AssessmentComponent component;

    @Column(name = "marks_obtained", precision = 5, scale = 2)
    private Double marksObtained;

    @Column(name = "marks_total", precision = 5, scale = 2)
    private Double marksTotal;

    @Column(name = "is_passed")
    private boolean isPassed = false;

    // ---- Getters & Setters ----

    public Integer getGradeId()                         { return gradeId; }
    public void setGradeId(Integer id)                  { this.gradeId = id; }
    public StudentCourse getEnrollment()                { return enrollment; }
    public void setEnrollment(StudentCourse e)          { this.enrollment = e; }
    public AssessmentComponent getComponent()           { return component; }
    public void setComponent(AssessmentComponent c)     { this.component = c; }
    public Double getMarksObtained()                    { return marksObtained; }
    public void setMarksObtained(Double m)              { this.marksObtained = m; }
    public Double getMarksTotal()                       { return marksTotal; }
    public void setMarksTotal(Double m)                 { this.marksTotal = m; }
    public boolean isPassed()                           { return isPassed; }
    public void setPassed(boolean passed)               { isPassed = passed; }
}
