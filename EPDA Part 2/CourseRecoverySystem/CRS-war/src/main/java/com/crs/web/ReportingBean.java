package com.crs.web;

import com.crs.ejb.EmailService;
import com.crs.ejb.ReportingService;
import com.crs.ejb.StudentService;
import com.crs.entity.Student;

import jakarta.ejb.EJB;
import jakarta.enterprise.context.SessionScoped;
import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.FacesContext;
import jakarta.inject.Named;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Named("reportingBean")
@SessionScoped
public class ReportingBean implements Serializable {

    @EJB private ReportingService reportingService;
    @EJB private StudentService   studentService;
    @EJB private EmailService     emailService;

    private List<Student>                           students;
    private int                                     selectedStudentId;
    private int                                     selectedSemester = 1;
    private int                                     selectedYear     = 1;
    private ReportingService.SemesterReport         semesterReport;
    private Map<String, ReportingService.SemesterReport> fullReport;

    public List<Student> getStudents() {
        if (students == null) students = studentService.findAll();
        return students;
    }

    public String generateSemesterReport() {
        if (selectedStudentId <= 0) { addError("Please select a student."); return null; }
        semesterReport = reportingService.generateSemesterReport(
                selectedStudentId, selectedSemester, selectedYear);
        fullReport = null;
        return null;
    }

    public String generateFullReport() {
        if (selectedStudentId <= 0) { addError("Please select a student."); return null; }
        fullReport = reportingService.generateFullReport(selectedStudentId);
        semesterReport = null;
        return null;
    }

    public String emailReport() {
        if (semesterReport == null) { addError("Generate a report first."); return null; }
        Student s = semesterReport.getStudent();
        emailService.sendPerformanceReport(s.getEmail(), s.getFullName(),
                semesterReport.getSemester(), semesterReport.getYearOfStudy());
        addInfo("Report emailed to " + s.getEmail());
        return null;
    }

    private void addInfo(String msg) {
        FacesContext.getCurrentInstance().addMessage(null,
                new FacesMessage(FacesMessage.SEVERITY_INFO, msg, null));
    }

    private void addError(String msg) {
        FacesContext.getCurrentInstance().addMessage(null,
                new FacesMessage(FacesMessage.SEVERITY_ERROR, msg, null));
    }

    /** Called via AJAX when the student dropdown changes — pre-fills year from the student's record. */
    public void onStudentChange() {
        if (selectedStudentId > 0) {
            getStudents().stream()
                .filter(s -> s.getStudentId() == selectedStudentId)
                .findFirst()
                .ifPresent(s -> selectedYear = s.getYearOfStudy());
        }
    }

    // Getters & Setters
    public int getSelectedStudentId()                   { return selectedStudentId; }
    public void setSelectedStudentId(int id)            { this.selectedStudentId = id; }
    public int getSelectedSemester()                    { return selectedSemester; }
    public void setSelectedSemester(int s)              { this.selectedSemester = s; }
    public int getSelectedYear()                        { return selectedYear; }
    public void setSelectedYear(int y)                  { this.selectedYear = y; }
    public ReportingService.SemesterReport getSemesterReport() { return semesterReport; }
    public Map<String, ReportingService.SemesterReport> getFullReport() { return fullReport; }

    /** Returns full report as an ordered list of entries for use in ui:repeat. */
    public List<Map.Entry<String, ReportingService.SemesterReport>> getFullReportEntries() {
        if (fullReport == null) return new ArrayList<>();
        return new ArrayList<>(fullReport.entrySet());
    }
}
