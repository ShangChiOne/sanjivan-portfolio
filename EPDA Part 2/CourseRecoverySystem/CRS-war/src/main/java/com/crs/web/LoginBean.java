package com.crs.web;

import com.crs.ejb.UserService;
import com.crs.entity.User;

import jakarta.ejb.EJB;
import jakarta.enterprise.context.RequestScoped;
import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.ExternalContext;
import jakarta.faces.context.FacesContext;
import jakarta.inject.Named;
import jakarta.servlet.http.HttpSession;
import java.io.IOException;
import java.io.Serializable;

@Named("loginBean")
@RequestScoped
public class LoginBean implements Serializable {

    @EJB
    private UserService userService;

    private String username;
    private String password;

    public String login() {
        User user = userService.login(username, password);
        if (user == null) {
            FacesContext.getCurrentInstance().addMessage(null,
                    new FacesMessage(FacesMessage.SEVERITY_ERROR,
                            "Invalid username or password.", null));
            return null;
        }

        ExternalContext ec = FacesContext.getCurrentInstance().getExternalContext();
        HttpSession session = (HttpSession) ec.getSession(true);
        session.setAttribute("currentUser", user);
        session.setAttribute("userRole", user.getRole().name());

        return "/dashboard.xhtml?faces-redirect=true";
    }

    public String logout() {
        ExternalContext ec = FacesContext.getCurrentInstance().getExternalContext();
        ((HttpSession) ec.getSession(false)).invalidate();
        return "/login.xhtml?faces-redirect=true";
    }

    // Getters & Setters
    public String getUsername()            { return username; }
    public void setUsername(String u)      { this.username = u; }
    public String getPassword()            { return password; }
    public void setPassword(String p)      { this.password = p; }
}
