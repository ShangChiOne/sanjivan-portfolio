package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;

@Entity
@Table(name = "recovery_failed_components")
@NamedQuery(name = "RecoveryFailedComponent.findByPlan",
            query = "SELECT r FROM RecoveryFailedComponent r WHERE r.plan.planId = :planId")
public class RecoveryFailedComponent implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Integer id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "plan_id", nullable = false)
    private RecoveryPlan plan;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "component_id", nullable = false)
    private AssessmentComponent component;

    @Column(name = "recovery_grade", precision = 5, scale = 2)
    private Double recoveryGrade;

    @Column(name = "is_recovered")
    private boolean isRecovered = false;

    // ---- Getters & Setters ----

    public Integer getId()                              { return id; }
    public void setId(Integer id)                       { this.id = id; }
    public RecoveryPlan getPlan()                       { return plan; }
    public void setPlan(RecoveryPlan plan)               { this.plan = plan; }
    public AssessmentComponent getComponent()           { return component; }
    public void setComponent(AssessmentComponent c)     { this.component = c; }
    public Double getRecoveryGrade()                    { return recoveryGrade; }
    public void setRecoveryGrade(Double g)              { this.recoveryGrade = g; }
    public boolean isRecovered()                        { return isRecovered; }
    public void setRecovered(boolean recovered)         { isRecovered = recovered; }
}
