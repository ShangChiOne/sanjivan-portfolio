package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "recovery_plans")
@NamedQueries({
    @NamedQuery(name = "RecoveryPlan.findAll",
                query = "SELECT r FROM RecoveryPlan r ORDER BY r.createdAt DESC"),
    @NamedQuery(name = "RecoveryPlan.findByStudent",
                query = "SELECT r FROM RecoveryPlan r WHERE r.student.studentId = :studentId ORDER BY r.createdAt DESC"),
    @NamedQuery(name = "RecoveryPlan.findActive",
                query = "SELECT r FROM RecoveryPlan r WHERE r.status = 'ACTIVE' ORDER BY r.createdAt DESC")
})
public class RecoveryPlan implements Serializable {

    public enum Status { ACTIVE, COMPLETED, FAILED }

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "plan_id")
    private Integer planId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "student_id", nullable = false)
    private Student student;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "enrollment_id", nullable = false)
    private StudentCourse enrollment;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "created_by", nullable = false)
    private User createdBy;

    @Enumerated(EnumType.STRING)
    @Column(name = "status")
    private Status status = Status.ACTIVE;

    @Column(name = "recommendation", columnDefinition = "TEXT")
    private String recommendation;

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @OneToMany(mappedBy = "plan", fetch = FetchType.LAZY, cascade = CascadeType.ALL, orphanRemoval = true)
    private List<RecoveryMilestone> milestones;

    @OneToMany(mappedBy = "plan", fetch = FetchType.LAZY, cascade = CascadeType.ALL, orphanRemoval = true)
    private List<RecoveryFailedComponent> failedComponents;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() { updatedAt = LocalDateTime.now(); }

    // ---- Getters & Setters ----

    public Integer getPlanId()                              { return planId; }
    public void setPlanId(Integer id)                       { this.planId = id; }
    public Student getStudent()                             { return student; }
    public void setStudent(Student s)                       { this.student = s; }
    public StudentCourse getEnrollment()                    { return enrollment; }
    public void setEnrollment(StudentCourse e)              { this.enrollment = e; }
    public User getCreatedBy()                              { return createdBy; }
    public void setCreatedBy(User u)                        { this.createdBy = u; }
    public Status getStatus()                               { return status; }
    public void setStatus(Status status)                    { this.status = status; }
    public String getRecommendation()                       { return recommendation; }
    public void setRecommendation(String r)                 { this.recommendation = r; }
    public LocalDateTime getCreatedAt()                     { return createdAt; }
    public LocalDateTime getUpdatedAt()                     { return updatedAt; }
    public List<RecoveryMilestone> getMilestones()          { return milestones; }
    public void setMilestones(List<RecoveryMilestone> m)    { this.milestones = m; }
    public List<RecoveryFailedComponent> getFailedComponents() { return failedComponents; }
    public void setFailedComponents(List<RecoveryFailedComponent> f) { this.failedComponents = f; }
}
