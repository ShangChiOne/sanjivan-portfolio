package com.crs.web;

import com.crs.ejb.EmailService;
import com.crs.ejb.UserService;
import com.crs.entity.User;

import jakarta.ejb.EJB;
import jakarta.enterprise.context.SessionScoped;
import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.FacesContext;
import jakarta.inject.Named;
import java.io.Serializable;
import java.util.List;
import java.util.UUID;

@Named("userManagementBean")
@SessionScoped
public class UserManagementBean implements Serializable {

    @EJB private UserService  userService;
    @EJB private EmailService emailService;

    private User        newUser      = new User();
    private String      newPassword;
    private User        selectedUser;
    private List<User>  users;

    // ---- Load ----

    public List<User> getUsers() {
        if (users == null) refresh();
        return users;
    }

    public void refresh() {
        users = userService.findAll();
    }

    // ---- Create ----

    public String createUser() {
        if (userService.usernameExists(newUser.getUsername())) {
            addError("Username already exists.");
            return null;
        }
        userService.createUser(newUser, newPassword);
        emailService.sendAccountCreated(newUser.getEmail(), newUser.getFullName(), newPassword);
        addInfo("User created successfully.");
        newUser = new User();
        newPassword = null;
        refresh();
        return null;
    }

    // ---- Update ----

    public void selectUser(User user) {
        this.selectedUser = user;
    }

    public String updateUser() {
        userService.updateUser(selectedUser);
        addInfo("User updated.");
        refresh();
        return null;
    }

    // ---- Deactivate ----

    public String deactivateUser(int userId) {
        User u = userService.findById(userId);
        userService.deactivateUser(userId);
        if (u != null) {
            emailService.sendAccountDeactivated(u.getEmail(), u.getFullName());
        }
        addInfo("User deactivated.");
        refresh();
        return null;
    }

    // ---- Password Reset ----

    public String resetPassword(int userId) {
        String newPass = "Temp@" + UUID.randomUUID().toString().substring(0, 6);
        userService.resetPassword(userId, newPass);
        User u = userService.findById(userId);
        emailService.sendPasswordReset(u.getEmail(), u.getFullName(), newPass);
        addInfo("Password reset. New password sent to " + u.getEmail());
        return null;
    }

    // ---- Helpers ----

    private void addError(String msg) {
        FacesContext.getCurrentInstance().addMessage(null,
                new FacesMessage(FacesMessage.SEVERITY_ERROR, msg, null));
    }

    private void addInfo(String msg) {
        FacesContext.getCurrentInstance().addMessage(null,
                new FacesMessage(FacesMessage.SEVERITY_INFO, msg, null));
    }

    // Getters & Setters
    public User getNewUser()                    { return newUser; }
    public void setNewUser(User u)              { this.newUser = u; }
    public String getNewPassword()              { return newPassword; }
    public void setNewPassword(String p)        { this.newPassword = p; }
    public User getSelectedUser()               { return selectedUser; }
    public void setSelectedUser(User u)         { this.selectedUser = u; }
}
