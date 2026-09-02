package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;

@Entity
@Table(name = "assessment_components")
@NamedQuery(name = "AssessmentComponent.findByCourse",
            query = "SELECT ac FROM AssessmentComponent ac WHERE ac.course.courseId = :courseId")
public class AssessmentComponent implements Serializable {

    public enum ComponentType { ASSIGNMENT, EXAM, PROJECT, QUIZ }

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "component_id")
    private Integer componentId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "course_id", nullable = false)
    private Course course;

    @Column(name = "component_name", nullable = false, length = 100)
    private String componentName;

    @Enumerated(EnumType.STRING)
    @Column(name = "component_type", nullable = false)
    private ComponentType componentType;

    @Column(name = "weightage", nullable = false, precision = 5, scale = 2)
    private Double weightage;

    // ---- Getters & Setters ----

    public Integer getComponentId()                         { return componentId; }
    public void setComponentId(Integer id)                  { this.componentId = id; }
    public Course getCourse()                               { return course; }
    public void setCourse(Course course)                    { this.course = course; }
    public String getComponentName()                        { return componentName; }
    public void setComponentName(String name)               { this.componentName = name; }
    public ComponentType getComponentType()                 { return componentType; }
    public void setComponentType(ComponentType type)        { this.componentType = type; }
    public Double getWeightage()                            { return weightage; }
    public void setWeightage(Double weightage)              { this.weightage = weightage; }
}
