package com.crs.web;

import com.crs.ejb.CourseService;
import com.crs.ejb.EmailService;
import com.crs.ejb.RecoveryService;
import com.crs.ejb.StudentService;
import com.crs.entity.*;
import com.crs.util.GradeUtil;

import jakarta.ejb.EJB;
import jakarta.enterprise.context.SessionScoped;
import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.FacesContext;
import jakarta.inject.Named;
import jakarta.servlet.http.HttpSession;
import java.io.Serializable;
import java.time.LocalDate;
import java.util.List;

@Named("recoveryPlanBean")
@SessionScoped
public class RecoveryPlanBean implements Serializable {

    @EJB private RecoveryService recoveryService;
    @EJB private StudentService  studentService;
    @EJB private CourseService   courseService;
    @EJB private EmailService    emailService;

    private List<RecoveryPlan> plans;
    private RecoveryPlan       selectedPlan;

    // New plan fields
    private int    selectedStudentId;
    private int    selectedEnrollmentId;
    private String recommendation;

    // Student / enrollment dropdowns for new plan dialog
    private List<Student>      students;
    private List<StudentCourse> failedEnrollments;

    // New milestone fields
    private String    studyWeek;
    private String    taskDescription;
    private LocalDate dueDate;

    // Edit recommendation fields
    private String    editRecommendation;

    // Edit milestone fields
    private int       selectedMilestoneId;
    private String    editStudyWeek;
    private String    editTaskDescription;
    private LocalDate editDueDate;

    // Failed component fields
    private List<AssessmentComponent> planComponents;
    private int    selectedComponentId;
    private int    selectedFailedComponentId;
    private String recoveryGrade;

    // ---- Plans ----

    public List<RecoveryPlan> getPlans() {
        if (plans == null) refresh();
        return plans;
    }

    public void refresh() {
        plans = recoveryService.findAllPlans();
    }

    public String createPlan() {
        HttpSession session = (HttpSession) FacesContext.getCurrentInstance()
                .getExternalContext().getSession(false);
        User currentUser = (User) session.getAttribute("currentUser");

        // Fetch enrollment and course title inside a service call (within transaction)
        // before creating the plan, to avoid accessing lazy proxies outside a transaction.
        StudentCourse enrollment = studentService.findEnrollmentById(selectedEnrollmentId);
        String courseTitle = enrollment.getCourse().getCourseTitle();

        recoveryService.createPlan(
                selectedStudentId, selectedEnrollmentId,
                currentUser.getUserId(), recommendation);

        Student student = studentService.findById(selectedStudentId);
        emailService.sendRecoveryPlanNotification(
                student.getEmail(), student.getFullName(), courseTitle);

        addInfo("Recovery plan created.");
        refresh();

        // Reset dialog fields
        selectedStudentId    = 0;
        selectedEnrollmentId = 0;
        recommendation       = null;
        failedEnrollments    = null;
        return null;
    }

    public void selectPlan(RecoveryPlan plan) {
        this.selectedPlan = plan;
        // Pre-populate edit recommendation field when plan is selected
        this.editRecommendation = plan.getRecommendation();
        // Load assessment components for this plan's course (avoids lazy loading)
        int courseId = recoveryService.getCourseIdForPlan(plan.getPlanId());
        this.planComponents = courseService.getComponents(courseId);
        this.selectedFailedComponentId = 0;
    }

    public String updateStatus(int planId, RecoveryPlan.Status status) {
        recoveryService.updatePlanStatus(planId, status);
        addInfo("Plan status updated to " + status);
        refresh();
        return null;
    }

    public String deletePlan(int planId) {
        recoveryService.deletePlan(planId);
        addInfo("Plan deleted.");
        refresh();
        return null;
    }

    // ---- Student / Enrollment dropdowns ----

    public List<Student> getStudents() {
        if (students == null) students = studentService.findAll();
        return students;
    }

    public void onStudentSelect() {
        if (selectedStudentId > 0) {
            failedEnrollments = studentService.getFailedCourses(selectedStudentId);
        } else {
            failedEnrollments = null;
        }
        selectedEnrollmentId = 0;
    }

    public List<StudentCourse> getFailedEnrollments() {
        return failedEnrollments;
    }

    // ---- Recommendation ----

    public void prepareEditRecommendation() {
        this.editRecommendation = selectedPlan != null ? selectedPlan.getRecommendation() : "";
    }

    public String updateRecommendation() {
        recoveryService.updateRecommendation(selectedPlan.getPlanId(), editRecommendation);
        selectedPlan.setRecommendation(editRecommendation);
        addInfo("Recommendation updated.");
        return null;
    }

    public String removeRecommendation() {
        recoveryService.updateRecommendation(selectedPlan.getPlanId(), null);
        selectedPlan.setRecommendation(null);
        editRecommendation = null;
        addInfo("Recommendation removed.");
        return null;
    }

    // ---- Milestones ----

    public String addMilestone() {
        recoveryService.addMilestone(selectedPlan.getPlanId(),
                studyWeek, taskDescription, dueDate);
        Student student = studentService.findById(selectedPlan.getStudent().getStudentId());
        emailService.sendMilestoneAdded(
                student.getEmail(), student.getFullName(), studyWeek, taskDescription);
        addInfo("Milestone added.");
        studyWeek = null; taskDescription = null; dueDate = null;
        return null;
    }

    public String completeMilestone(int milestoneId) {
        recoveryService.completeMilestone(milestoneId);
        addInfo("Milestone marked complete.");
        return null;
    }

    public String deleteMilestone(int milestoneId) {
        recoveryService.deleteMilestone(milestoneId);
        addInfo("Milestone removed.");
        return null;
    }

    public void selectMilestoneForEdit(RecoveryMilestone m) {
        this.selectedMilestoneId  = m.getMilestoneId();
        this.editStudyWeek        = m.getStudyWeek();
        this.editTaskDescription  = m.getTaskDescription();
        this.editDueDate          = m.getDueDate();
    }

    public String updateMilestone() {
        recoveryService.updateMilestoneDetails(
                selectedMilestoneId, editStudyWeek, editTaskDescription, editDueDate);
        addInfo("Milestone updated.");
        selectedMilestoneId = 0;
        return null;
    }

    public String cancelMilestoneEdit() {
        selectedMilestoneId = 0;
        return null;
    }

    public List<RecoveryMilestone> getMilestones() {
        return selectedPlan == null ? null
                : recoveryService.getMilestones(selectedPlan.getPlanId());
    }

    // ---- Failed Components ----

    public List<RecoveryFailedComponent> getFailedComponents() {
        return selectedPlan == null ? null
                : recoveryService.getFailedComponents(selectedPlan.getPlanId());
    }

    public List<AssessmentComponent> getPlanComponents() {
        return planComponents;
    }

    public String addFailedComponent() {
        if (selectedPlan == null || selectedComponentId == 0) return null;
        recoveryService.addFailedComponent(selectedPlan.getPlanId(), selectedComponentId);
        addInfo("Failed component added.");
        selectedComponentId = 0;
        return null;
    }

    public String deleteFailedComponent(int failedComponentId) {
        recoveryService.deleteFailedComponent(failedComponentId);
        addInfo("Failed component removed.");
        return null;
    }

    public void selectFailedComponent(int failedComponentId) {
        this.selectedFailedComponentId = failedComponentId;
        this.recoveryGrade = null;
    }

    public String recordRecoveryGrade() {
        if (recoveryGrade == null || recoveryGrade.isEmpty()) {
            addError("Please select a grade.");
            return null;
        }
        double gradePoint = GradeUtil.toGradePoint(recoveryGrade);
        recoveryService.recordRecoveryGrade(selectedFailedComponentId, gradePoint);
        addInfo("Recovery grade recorded.");
        selectedFailedComponentId = 0;
        recoveryGrade = null;
        return null;
    }

    // ---- Helpers ----

    private void addInfo(String msg) {
        FacesContext.getCurrentInstance().addMessage(null,
                new FacesMessage(FacesMessage.SEVERITY_INFO, msg, null));
    }

    private void addError(String msg) {
        FacesContext.getCurrentInstance().addMessage(null,
                new FacesMessage(FacesMessage.SEVERITY_ERROR, msg, null));
    }

    // Getters & Setters
    public RecoveryPlan getSelectedPlan()                   { return selectedPlan; }
    public void setSelectedPlan(RecoveryPlan p)             { this.selectedPlan = p; }
    public int getSelectedStudentId()                       { return selectedStudentId; }
    public void setSelectedStudentId(int id)                { this.selectedStudentId = id; }
    public int getSelectedEnrollmentId()                    { return selectedEnrollmentId; }
    public void setSelectedEnrollmentId(int id)             { this.selectedEnrollmentId = id; }
    public String getRecommendation()                       { return recommendation; }
    public void setRecommendation(String r)                 { this.recommendation = r; }
    public String getStudyWeek()                            { return studyWeek; }
    public void setStudyWeek(String s)                      { this.studyWeek = s; }
    public String getTaskDescription()                      { return taskDescription; }
    public void setTaskDescription(String d)                { this.taskDescription = d; }
    public LocalDate getDueDate()                           { return dueDate; }
    public void setDueDate(LocalDate d)                     { this.dueDate = d; }
    public int getSelectedComponentId()                     { return selectedComponentId; }
    public void setSelectedComponentId(int id)              { this.selectedComponentId = id; }
    public int getSelectedFailedComponentId()               { return selectedFailedComponentId; }
    public String getRecoveryGrade()                        { return recoveryGrade; }
    public void setRecoveryGrade(String g)                  { this.recoveryGrade = g; }
    public String getEditRecommendation()                   { return editRecommendation; }
    public void setEditRecommendation(String r)             { this.editRecommendation = r; }
    public int getSelectedMilestoneId()                     { return selectedMilestoneId; }
    public String getEditStudyWeek()                        { return editStudyWeek; }
    public void setEditStudyWeek(String s)                  { this.editStudyWeek = s; }
    public String getEditTaskDescription()                  { return editTaskDescription; }
    public void setEditTaskDescription(String d)            { this.editTaskDescription = d; }
    public LocalDate getEditDueDate()                       { return editDueDate; }
    public void setEditDueDate(LocalDate d)                 { this.editDueDate = d; }
}
