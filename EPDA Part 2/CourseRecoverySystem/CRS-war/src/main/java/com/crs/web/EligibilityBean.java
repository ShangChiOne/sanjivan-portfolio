package com.crs.web;

import com.crs.ejb.EligibilityService;
import com.crs.ejb.EmailService;
import com.crs.entity.Student;

import jakarta.ejb.EJB;
import jakarta.enterprise.context.SessionScoped;
import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.FacesContext;
import jakarta.inject.Named;
import java.io.Serializable;
import java.util.List;

@Named("eligibilityBean")
@SessionScoped
public class EligibilityBean implements Serializable {

    @EJB private EligibilityService eligibilityService;
    @EJB private EmailService       emailService;

    private List<Student> ineligibleStudents;
    private List<Student> eligibleStudents;
    private Student       selectedStudent;
    private double        cgpa;
    private long          failedCount;
    private int           confirmStudentId;
    private String        confirmStudentName;

    // ---- Load lists ----

    public List<Student> getIneligibleStudents() {
        if (ineligibleStudents == null) loadLists();
        return ineligibleStudents;
    }

    public List<Student> getEligibleStudents() {
        if (eligibleStudents == null) loadLists();
        return eligibleStudents;
    }

    public void loadLists() {
        ineligibleStudents = eligibilityService.getIneligibleStudents();
        eligibleStudents   = eligibilityService.getEligibleStudents();
    }

    // ---- Check individual student ----

    public void checkStudent(Student student) {
        this.selectedStudent = student;
        this.cgpa        = eligibilityService.calculateCGPA(student.getStudentId());
        this.failedCount = eligibilityService.countFailedCourses(student.getStudentId());
    }

    // ---- Prepare confirmation dialog ----

    public void prepareProgress(Student s) {
        this.confirmStudentId   = s.getStudentId();
        this.confirmStudentName = s.getFullName();
    }

    // ---- Progress to next year ----

    public String progressStudent(int studentId) {
        // Capture details before lists are refreshed
        Student found = eligibleStudents == null ? null :
            eligibleStudents.stream()
                .filter(st -> st.getStudentId() == studentId)
                .findFirst().orElse(null);
        try {
            boolean graduated = eligibilityService.progressToNextYear(studentId);
            loadLists();
            if (found != null) {
                emailService.sendEligibilityResult(found.getEmail(), found.getFullName(), true);
                if (graduated) {
                    addInfo(found.getFullName() + " has graduated successfully!");
                } else {
                    addInfo(found.getFullName() + " progressed to Year " + (found.getYearOfStudy() + 1) + ".");
                }
            } else {
                addInfo("Student progressed successfully.");
            }
        } catch (IllegalStateException e) {
            addError(e.getMessage());
        }
        return null;
    }

    // ---- Notify ineligible student ----

    public String notifyIneligible(Student student) {
        emailService.sendEligibilityResult(student.getEmail(), student.getFullName(), false);
        addInfo("Notification sent to " + student.getEmail());
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
    public Student getSelectedStudent()     { return selectedStudent; }
    public double getCgpa()                 { return cgpa; }
    public long getFailedCount()            { return failedCount; }
    public int getConfirmStudentId()        { return confirmStudentId; }
    public String getConfirmStudentName()   { return confirmStudentName; }
}
