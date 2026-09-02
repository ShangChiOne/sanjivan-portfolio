package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;
import java.util.List;

@Entity
@Table(name = "courses")
@NamedQueries({
    @NamedQuery(name = "Course.findAll",
                query = "SELECT c FROM Course c WHERE c.isActive = true ORDER BY c.courseCode"),
    @NamedQuery(name = "Course.findByCode",
                query = "SELECT c FROM Course c WHERE c.courseCode = :courseCode")
})
public class Course implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "course_id")
    private Integer courseId;

    @Column(name = "course_code", nullable = false, unique = true, length = 20)
    private String courseCode;

    @Column(name = "course_title", nullable = false, length = 100)
    private String courseTitle;

    @Column(name = "credit_hours", nullable = false)
    private int creditHours;

    @Column(name = "is_active")
    private boolean isActive = true;

    @OneToMany(mappedBy = "course", fetch = FetchType.LAZY, cascade = CascadeType.ALL)
    private List<AssessmentComponent> components;

    @OneToMany(mappedBy = "course", fetch = FetchType.LAZY)
    private List<StudentCourse> enrollments;

    // ---- Getters & Setters ----

    public Integer getCourseId()                        { return courseId; }
    public void setCourseId(Integer courseId)           { this.courseId = courseId; }
    public String getCourseCode()                       { return courseCode; }
    public void setCourseCode(String courseCode)        { this.courseCode = courseCode; }
    public String getCourseTitle()                      { return courseTitle; }
    public void setCourseTitle(String courseTitle)      { this.courseTitle = courseTitle; }
    public int getCreditHours()                         { return creditHours; }
    public void setCreditHours(int creditHours)         { this.creditHours = creditHours; }
    public boolean isActive()                           { return isActive; }
    public void setActive(boolean active)               { isActive = active; }
    public List<AssessmentComponent> getComponents()    { return components; }
    public void setComponents(List<AssessmentComponent> c) { this.components = c; }
    public List<StudentCourse> getEnrollments()         { return enrollments; }
    public void setEnrollments(List<StudentCourse> e)   { this.enrollments = e; }
}
