package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;
import java.time.LocalDate;

@Entity
@Table(name = "recovery_milestones")
@NamedQuery(name = "RecoveryMilestone.findByPlan",
            query = "SELECT m FROM RecoveryMilestone m WHERE m.plan.planId = :planId ORDER BY m.dueDate")
public class RecoveryMilestone implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "milestone_id")
    private Integer milestoneId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "plan_id", nullable = false)
    private RecoveryPlan plan;

    @Column(name = "study_week", nullable = false, length = 50)
    private String studyWeek;

    @Column(name = "task_description", nullable = false, length = 255)
    private String taskDescription;

    @Column(name = "due_date")
    private LocalDate dueDate;

    @Column(name = "is_completed")
    private boolean isCompleted = false;

    // ---- Getters & Setters ----

    public Integer getMilestoneId()                 { return milestoneId; }
    public void setMilestoneId(Integer id)          { this.milestoneId = id; }
    public RecoveryPlan getPlan()                   { return plan; }
    public void setPlan(RecoveryPlan plan)           { this.plan = plan; }
    public String getStudyWeek()                    { return studyWeek; }
    public void setStudyWeek(String studyWeek)      { this.studyWeek = studyWeek; }
    public String getTaskDescription()              { return taskDescription; }
    public void setTaskDescription(String d)        { this.taskDescription = d; }
    public LocalDate getDueDate()                   { return dueDate; }
    public void setDueDate(LocalDate dueDate)       { this.dueDate = dueDate; }
    public boolean isCompleted()                    { return isCompleted; }
    public void setCompleted(boolean completed)     { isCompleted = completed; }
}
