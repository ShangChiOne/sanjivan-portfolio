package com.crs.filter;

import com.crs.entity.User;
import jakarta.servlet.*;
import jakarta.servlet.annotation.WebFilter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import java.io.IOException;

/**
 * Redirects unauthenticated users to the login page.
 * Pages excluded from the filter: login.xhtml, index.xhtml.
 */
@WebFilter("*.xhtml")
public class AuthFilter implements Filter {

    private static final String LOGIN_PAGE = "/login.xhtml";

    @Override
    public void doFilter(ServletRequest request, ServletResponse response,
                         FilterChain chain) throws IOException, ServletException {

        HttpServletRequest  req  = (HttpServletRequest)  request;
        HttpServletResponse resp = (HttpServletResponse) response;

        String uri = req.getRequestURI();

        // Allow static resources, login page, and index (redirect) through
        if (uri.contains(LOGIN_PAGE)
                || uri.contains("/index.xhtml")
                || uri.contains("/jakarta.faces.resource")) {
            chain.doFilter(request, response);
            return;
        }

        HttpSession session = req.getSession(false);
        User currentUser = (session != null) ? (User) session.getAttribute("currentUser") : null;

        if (currentUser == null) {
            resp.sendRedirect(req.getContextPath() + LOGIN_PAGE);
            return;
        }

        // Course Recovery Plans is restricted to Academic Officers only
        String role = (String) session.getAttribute("userRole");
        if (uri.contains("/recovery-plan.xhtml") && !"ACADEMIC_OFFICER".equals(role)) {
            resp.sendRedirect(req.getContextPath() + "/dashboard.xhtml");
            return;
        }

        chain.doFilter(request, response);
    }
}
