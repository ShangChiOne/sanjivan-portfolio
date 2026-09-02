package com.crs.web;

import com.crs.ejb.CourseService;
import com.crs.ejb.StudentService;
import com.crs.entity.Course;
import com.crs.entity.Student;
import com.crs.entity.StudentCourse;

import jakarta.ejb.EJB;
import jakarta.enterprise.context.SessionScoped;
import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.FacesContext;
import jakarta.inject.Named;
import java.io.Serializable;
import java.util.List;

@Named("studentManagementBean")
@SessionScoped
public class StudentManagementBean implements Serializable {

    @EJB private StudentService studentService;
    @EJB private CourseService  courseService;

    private Student             newStudent          = new Student();
    private Student             selectedStudent;
    private List<Student>       students;
    private List<StudentCourse> selectedEnrollments;

    // Enrollment form fields
    private int enrollCourseId;
    private int enrollSemester  = 1;
    private int enrollYear      = 1;
    private int enrollAttempt   = 1;

    // Grade recording fields
    private int    gradeEnrollmentId;
    private String newGrade;

    // ---- Load ----

    public List<Student> getStudents() {
        if (students == null) refresh();
        return students;
    }

    public List<Course> getCourses() {
        return courseService.findAll();
    }

    public void refresh() {
        students = studentService.findAll();
    }

    // ---- Create Student ----

    public String createStudent() {
        studentService.createStudent(newStudent);
        addInfo("Student " + newStudent.getStudentNumber() + " added.");
        newStudent = new Student();
        refresh();
        return null;
    }

    // ---- Select / Update ----

    public void selectStudent(Student s) {
        this.selectedStudent   = s;
        this.selectedEnrollments = studentService.getEnrollments(s.getStudentId());
    }

    public String updateStudent() {
        studentService.updateStudent(selectedStudent);
        addInfo("Student updated.");
        refresh();
        return null;
    }

    // ---- Deactivate ----

    public String deactivateStudent(int studentId) {
        studentService.deactivateStudent(studentId);
        addInfo("Student deactivated.");
        refresh();
        return null;
    }

    // ---- Enrol in Course ----

    public String enrolStudent() {
        if (selectedStudent == null) return null;
        try {
            courseService.enrol(selectedStudent.getStudentId(),
                    enrollCourseId, enrollSemester, enrollYear, enrollAttempt);
            addInfo("Student enrolled successfully.");
            selectedEnrollments = studentService.getEnrollments(selectedStudent.getStudentId());
        } catch (Exception e) {
            addError("Enrollment failed: " + e.getMessage());
        }
        return null;
    }

    // ---- Grade Recording ----

    public void selectEnrollment(int enrollmentId) {
        this.gradeEnrollmentId = enrollmentId;
    }

    public String recordGrade() {
        try {
            courseService.recordGrade(gradeEnrollmentId, newGrade);
            addInfo("Grade recorded.");
            if (selectedStudent != null) {
                selectedEnrollments = studentService.getEnrollments(selectedStudent.getStudentId());
            }
        } catch (Exception e) {
            addError("Failed to record grade: " + e.getMessage());
        }
        newGrade = null;
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
    public Student getNewStudent()                          { return newStudent; }
    public void setNewStudent(Student s)                   { this.newStudent = s; }
    public Student getSelectedStudent()                    { return selectedStudent; }
    public void setSelectedStudent(Student s)              { this.selectedStudent = s; }
    public List<StudentCourse> getSelectedEnrollments()    { return selectedEnrollments; }
    public int getEnrollCourseId()                         { return enrollCourseId; }
    public void setEnrollCourseId(int id)                  { this.enrollCourseId = id; }
    public int getEnrollSemester()                         { return enrollSemester; }
    public void setEnrollSemester(int s)                   { this.enrollSemester = s; }
    public int getEnrollYear()                             { return enrollYear; }
    public void setEnrollYear(int y)                       { this.enrollYear = y; }
    public int getEnrollAttempt()                          { return enrollAttempt; }
    public void setEnrollAttempt(int a)                    { this.enrollAttempt = a; }
    public int getGradeEnrollmentId()                      { return gradeEnrollmentId; }
    public void setGradeEnrollmentId(int id)               { this.gradeEnrollmentId = id; }
    public String getNewGrade()                            { return newGrade; }
    public void setNewGrade(String g)                      { this.newGrade = g; }
}
