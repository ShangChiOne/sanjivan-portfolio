package com.crs.ejb;

import com.crs.entity.*;

import jakarta.ejb.Stateless;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.TypedQuery;
import java.util.List;

@Stateless
public class RecoveryService {

    /** Minimum grade point (GPA scale 0.0–4.0) required to be considered recovered. */
    private static final double COMPONENT_PASSING_MARK = 2.0;

    @PersistenceContext(unitName = "CRSPU")
    private EntityManager em;

    // ---- Recovery Plans ----

    public RecoveryPlan createPlan(int studentId, int enrollmentId,
                                   int createdByUserId, String recommendation) {
        RecoveryPlan plan = new RecoveryPlan();
        plan.setStudent(em.getReference(Student.class, studentId));
        plan.setEnrollment(em.getReference(StudentCourse.class, enrollmentId));
        plan.setCreatedBy(em.getReference(User.class, createdByUserId));
        plan.setRecommendation(recommendation);
        plan.setStatus(RecoveryPlan.Status.ACTIVE);
        em.persist(plan);
        return plan;
    }

    public RecoveryPlan findPlanById(int planId) {
        return em.find(RecoveryPlan.class, planId);
    }

    public List<RecoveryPlan> findAllPlans() {
        return em.createNamedQuery("RecoveryPlan.findAll", RecoveryPlan.class).getResultList();
    }

    public List<RecoveryPlan> findPlansByStudent(int studentId) {
        TypedQuery<RecoveryPlan> q = em.createNamedQuery("RecoveryPlan.findByStudent", RecoveryPlan.class);
        q.setParameter("studentId", studentId);
        return q.getResultList();
    }

    public List<RecoveryPlan> findActivePlans() {
        return em.createNamedQuery("RecoveryPlan.findActive", RecoveryPlan.class).getResultList();
    }

    public void updatePlan(RecoveryPlan plan) {
        em.merge(plan);
    }

    public void updatePlanStatus(int planId, RecoveryPlan.Status status) {
        RecoveryPlan p = findPlanById(planId);
        if (p != null) { p.setStatus(status); em.merge(p); }
    }

    public void updateRecommendation(int planId, String recommendation) {
        RecoveryPlan p = findPlanById(planId);
        if (p != null) { p.setRecommendation(recommendation); em.merge(p); }
    }

    public void deletePlan(int planId) {
        RecoveryPlan p = findPlanById(planId);
        if (p != null) em.remove(p);
    }

    // ---- Milestones ----

    public RecoveryMilestone addMilestone(int planId, String studyWeek,
                                          String taskDescription,
                                          java.time.LocalDate dueDate) {
        RecoveryMilestone m = new RecoveryMilestone();
        m.setPlan(em.getReference(RecoveryPlan.class, planId));
        m.setStudyWeek(studyWeek);
        m.setTaskDescription(taskDescription);
        m.setDueDate(dueDate);
        em.persist(m);
        return m;
    }

    public void updateMilestone(RecoveryMilestone milestone) {
        em.merge(milestone);
    }

    public void updateMilestoneDetails(int milestoneId, String studyWeek,
                                        String taskDescription, java.time.LocalDate dueDate) {
        RecoveryMilestone m = em.find(RecoveryMilestone.class, milestoneId);
        if (m != null) {
            m.setStudyWeek(studyWeek);
            m.setTaskDescription(taskDescription);
            m.setDueDate(dueDate);
            em.merge(m);
        }
    }

    public void completeMilestone(int milestoneId) {
        RecoveryMilestone m = em.find(RecoveryMilestone.class, milestoneId);
        if (m != null) { m.setCompleted(true); em.merge(m); }
    }

    public void deleteMilestone(int milestoneId) {
        RecoveryMilestone m = em.find(RecoveryMilestone.class, milestoneId);
        if (m != null) em.remove(m);
    }

    public List<RecoveryMilestone> getMilestones(int planId) {
        TypedQuery<RecoveryMilestone> q =
                em.createNamedQuery("RecoveryMilestone.findByPlan", RecoveryMilestone.class);
        q.setParameter("planId", planId);
        return q.getResultList();
    }

    // ---- Failed Components ----

    public void addFailedComponent(int planId, int componentId) {
        RecoveryFailedComponent rfc = new RecoveryFailedComponent();
        rfc.setPlan(em.getReference(RecoveryPlan.class, planId));
        rfc.setComponent(em.getReference(AssessmentComponent.class, componentId));
        em.persist(rfc);
    }

    public void deleteFailedComponent(int failedComponentId) {
        RecoveryFailedComponent rfc = em.find(RecoveryFailedComponent.class, failedComponentId);
        if (rfc != null) em.remove(rfc);
    }

    public void recordRecoveryGrade(int failedComponentId, double grade) {
        RecoveryFailedComponent rfc = em.find(RecoveryFailedComponent.class, failedComponentId);
        if (rfc != null) {
            rfc.setRecoveryGrade(grade);
            rfc.setRecovered(grade >= COMPONENT_PASSING_MARK);
            em.merge(rfc);
        }
    }

    /** Returns the courseId linked to a plan's enrollment (avoids lazy-loading in web tier). */
    public int getCourseIdForPlan(int planId) {
        return em.createQuery(
                "SELECT sc.course.courseId FROM RecoveryPlan rp JOIN rp.enrollment sc WHERE rp.planId = :planId",
                Integer.class)
                .setParameter("planId", planId)
                .getSingleResult();
    }

    public List<RecoveryFailedComponent> getFailedComponents(int planId) {
        TypedQuery<RecoveryFailedComponent> q =
                em.createNamedQuery("RecoveryFailedComponent.findByPlan", RecoveryFailedComponent.class);
        q.setParameter("planId", planId);
        return q.getResultList();
    }
}
