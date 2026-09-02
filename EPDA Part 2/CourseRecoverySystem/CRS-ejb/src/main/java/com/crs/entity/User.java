package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;
import java.time.LocalDateTime;

@Entity
@Table(name = "users")
@NamedQueries({
    @NamedQuery(name = "User.findAll",
                query = "SELECT u FROM User u"),
    @NamedQuery(name = "User.findByUsername",
                query = "SELECT u FROM User u WHERE u.username = :username"),
    @NamedQuery(name = "User.findByEmail",
                query = "SELECT u FROM User u WHERE u.email = :email"),
    @NamedQuery(name = "User.authenticate",
                query = "SELECT u FROM User u WHERE u.username = :username AND u.password = :password AND u.isActive = true")
})
public class User implements Serializable {

    public enum Role { COURSE_ADMIN, ACADEMIC_OFFICER }

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_id")
    private Integer userId;

    @Column(name = "username", nullable = false, unique = true, length = 50)
    private String username;

    @Column(name = "password", nullable = false, length = 255)
    private String password;

    @Column(name = "email", nullable = false, unique = true, length = 100)
    private String email;

    @Column(name = "full_name", nullable = false, length = 100)
    private String fullName;

    @Enumerated(EnumType.STRING)
    @Column(name = "role", nullable = false)
    private Role role;

    @Column(name = "is_active")
    private boolean isActive = true;

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // ---- Getters & Setters ----

    public Integer getUserId()                  { return userId; }
    public void setUserId(Integer userId)       { this.userId = userId; }
    public String getUsername()                 { return username; }
    public void setUsername(String username)    { this.username = username; }
    public String getPassword()                 { return password; }
    public void setPassword(String password)    { this.password = password; }
    public String getEmail()                    { return email; }
    public void setEmail(String email)          { this.email = email; }
    public String getFullName()                 { return fullName; }
    public void setFullName(String fullName)    { this.fullName = fullName; }
    public Role getRole()                       { return role; }
    public void setRole(Role role)              { this.role = role; }
    public boolean isActive()                   { return isActive; }
    public void setActive(boolean active)       { isActive = active; }
    public LocalDateTime getCreatedAt()         { return createdAt; }
    public LocalDateTime getUpdatedAt()         { return updatedAt; }
}
