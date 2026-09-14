// Mobile nav toggle
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');

navToggle.addEventListener('click', () => {
  const isOpen = navLinks.classList.toggle('is-open');
  navToggle.setAttribute('aria-expanded', String(isOpen));
});

navLinks.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    navLinks.classList.remove('is-open');
    navToggle.setAttribute('aria-expanded', 'false');
  });
});

// Footer year
document.getElementById('year').textContent = new Date().getFullYear();

// Video modal
const modal = document.getElementById('videoModal');
const modalFrame = document.getElementById('videoModalFrame');

function openVideo(id) {
  modalFrame.innerHTML =
    `<iframe src="https://www.youtube-nocookie.com/embed/${id}?autoplay=1&rel=0" ` +
    `title="Project demo video" allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>`;
  modal.hidden = false;
  document.body.style.overflow = 'hidden';
}

function openImage(src, alt) {
  modalFrame.innerHTML = '';
  const img = document.createElement('img');
  img.src = src;
  img.alt = alt || '';
  img.style.width = '100%';
  img.style.height = '100%';
  img.style.objectFit = 'contain';
  img.style.borderRadius = '12px';
  modalFrame.appendChild(img);
  modal.hidden = false;
  document.body.style.overflow = 'hidden';
}

function closeVideo() {
  modal.hidden = true;
  modalFrame.innerHTML = '';
  document.body.style.overflow = '';
}

document.querySelectorAll('[data-video]').forEach((btn) => {
  btn.addEventListener('click', () => openVideo(btn.dataset.video));
});

document.querySelectorAll('[data-lightbox]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const img = btn.querySelector('img');
    openImage(btn.dataset.lightbox, img ? img.alt : '');
  });
});

modal.querySelectorAll('[data-close]').forEach((el) => {
  el.addEventListener('click', closeVideo);
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !modal.hidden) closeVideo();
});

// Theme toggle
const themeToggle = document.getElementById('themeToggle');
themeToggle.addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
});

// Scroll progress bar
const scrollProgress = document.getElementById('scrollProgress');
let progressTicking = false;
function updateScrollProgress() {
  const html = document.documentElement;
  const scrollable = html.scrollHeight - html.clientHeight;
  const pct = scrollable > 0 ? (html.scrollTop / scrollable) * 100 : 0;
  scrollProgress.style.width = pct + '%';
  progressTicking = false;
}
window.addEventListener('scroll', () => {
  if (!progressTicking) {
    requestAnimationFrame(updateScrollProgress);
    progressTicking = true;
  }
}, { passive: true });
updateScrollProgress();

// Scrollspy nav underline
const spySections = ['about', 'projects', 'contact']
  .map((id) => document.getElementById(id))
  .filter(Boolean);
const spyLinkMap = new Map();
navLinks.querySelectorAll('a[href^="#"]').forEach((a) => {
  spyLinkMap.set(a.getAttribute('href').slice(1), a);
});
const spyObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const link = spyLinkMap.get(entry.target.id);
      if (!link) return;
      spyLinkMap.forEach((a) => a.classList.remove('is-active'));
      link.classList.add('is-active');
    });
  },
  { rootMargin: '-45% 0px -45% 0px' }
);
spySections.forEach((s) => spyObserver.observe(s));

// Hero spotlight follows the cursor
const hero = document.querySelector('.hero');
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const canHover = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
if (hero && !prefersReducedMotion) {
  hero.addEventListener('mousemove', (e) => {
    const rect = hero.getBoundingClientRect();
    hero.style.setProperty('--spot-x', ((e.clientX - rect.left) / rect.width) * 100 + '%');
    hero.style.setProperty('--spot-y', ((e.clientY - rect.top) / rect.height) * 100 + '%');
  });
}

// Animated stat counters
function animateCounter(el) {
  const parts = el.dataset.count.split(',').map(Number);
  const decimals = parseInt(el.dataset.decimals || '0', 10);
  const suffix = el.dataset.suffix || '';
  const duration = 1100;
  const start = performance.now();
  function frame(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = parts.map((n) => (n * eased).toFixed(decimals)).join(' / ') + suffix;
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
const counters = document.querySelectorAll('[data-count]');
if (counters.length) {
  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          counterObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.6 }
  );
  counters.forEach((el) => counterObserver.observe(el));
}

// Tilt + glare on project cards
if (canHover && !prefersReducedMotion) {
  document.querySelectorAll('.card').forEach((card) => {
    const glare = document.createElement('div');
    glare.className = 'tilt-glare';
    card.appendChild(glare);

    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      const rx = (0.5 - y) * 10;
      const ry = (x - 0.5) * 10;
      card.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-4px)`;
      glare.style.background = `radial-gradient(circle at ${x * 100}% ${y * 100}%, rgba(255,255,255,0.35), transparent 55%)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
      glare.style.background = 'transparent';
    });
  });

  // Magnetic buttons
  document.querySelectorAll('.btn, .nav-cta').forEach((btn) => {
    btn.addEventListener('mousemove', (e) => {
      const rect = btn.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      btn.style.transform = `translate(${Math.max(-8, Math.min(8, x * 0.3))}px, ${Math.max(-8, Math.min(8, y * 0.3))}px)`;
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.transform = '';
    });
  });
}

// Toast + copy to clipboard
const toast = document.getElementById('toast');
let toastTimer;
function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2200);
}
function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  const area = document.createElement('textarea');
  area.value = text;
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.appendChild(area);
  area.select();
  document.execCommand('copy');
  document.body.removeChild(area);
  return Promise.resolve();
}
document.querySelectorAll('.copy-btn').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    const timeout = new Promise((_, reject) => setTimeout(reject, 1200));
    Promise.race([copyText(btn.dataset.copy), timeout])
      .then(() => showToast('Email copied to clipboard'))
      .catch(() => showToast('Could not copy — email is above'));
  });
});

// Command palette
const cmdk = document.getElementById('cmdk');
const cmdkInput = document.getElementById('cmdkInput');
const cmdkList = document.getElementById('cmdkList');
const cmdkTrigger = document.getElementById('cmdkTrigger');

const commands = [
  { label: 'About', hint: 'Section', action: () => scrollToId('about') },
  { label: 'Projects', hint: 'Section', action: () => scrollToId('projects') },
  { label: 'Contact', hint: 'Section', action: () => scrollToId('contact') },
  { label: 'ForecastIQ — Final Year Project', hint: 'Project', action: () => scrollToId('projects') },
  { label: 'SavePlus+', hint: 'Project', action: () => scrollToId('projects') },
  { label: 'Stock Portfolio Analyzer & Backtester', hint: 'Open site', action: () => window.open('https://stock-portfolio-backtester.vercel.app', '_blank', 'noopener') },
  { label: "Dragon's Ring Kickboxing Tournament", hint: 'Project', action: () => scrollToId('projects') },
  { label: 'Insurance Premium Prediction', hint: 'Project', action: () => scrollToId('projects') },
  { label: 'Thiyages & Co Law Firm', hint: 'Open site', action: () => window.open('https://thiyagesco.com', '_blank', 'noopener') },
  { label: 'Credit Risk Classification', hint: 'Project', action: () => scrollToId('projects') },
  { label: 'POTS', hint: 'Project', action: () => scrollToId('projects') },
  { label: 'Open GitHub', hint: 'Link', action: () => window.open('https://github.com/ShangChiOne', '_blank', 'noopener') },
  { label: 'Open LinkedIn', hint: 'Link', action: () => window.open('https://linkedin.com/in/sanjivan-thiyageswaran-660641279', '_blank', 'noopener') },
  { label: 'Email Sanjivan', hint: 'Contact', action: () => { window.location.href = 'mailto:sanjivanthiyages@gmail.com'; } },
  { label: 'Toggle dark mode', hint: 'Preference', action: () => themeToggle.click() },
];

function scrollToId(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth' });
}

let cmdkActiveIndex = 0;
let cmdkFiltered = commands;

function renderCmdk() {
  cmdkList.innerHTML = '';
  if (!cmdkFiltered.length) {
    cmdkList.innerHTML = '<li class="cmdk-empty">No matches</li>';
    return;
  }
  cmdkFiltered.forEach((cmd, i) => {
    const li = document.createElement('li');
    li.className = 'cmdk-item' + (i === cmdkActiveIndex ? ' is-active' : '');
    li.innerHTML = `<span>${cmd.label}</span><span class="cmdk-hint">${cmd.hint}</span>`;
    li.addEventListener('mouseenter', () => {
      cmdkActiveIndex = i;
      renderCmdk();
    });
    li.addEventListener('click', () => runCmdk(i));
    cmdkList.appendChild(li);
  });
}

function runCmdk(i) {
  const cmd = cmdkFiltered[i];
  if (!cmd) return;
  closeCmdk();
  cmd.action();
}

function openCmdk() {
  cmdk.hidden = false;
  document.body.style.overflow = 'hidden';
  cmdkInput.value = '';
  cmdkFiltered = commands;
  cmdkActiveIndex = 0;
  renderCmdk();
  setTimeout(() => cmdkInput.focus(), 0);
}

function closeCmdk() {
  cmdk.hidden = true;
  document.body.style.overflow = '';
  cmdkTrigger.focus();
}

cmdkTrigger.addEventListener('click', openCmdk);
cmdk.querySelectorAll('[data-close-cmdk]').forEach((el) => el.addEventListener('click', closeCmdk));

cmdkInput.addEventListener('input', () => {
  const q = cmdkInput.value.trim().toLowerCase();
  cmdkFiltered = q ? commands.filter((c) => c.label.toLowerCase().includes(q)) : commands;
  cmdkActiveIndex = 0;
  renderCmdk();
});

cmdkInput.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    cmdkActiveIndex = Math.min(cmdkActiveIndex + 1, cmdkFiltered.length - 1);
    renderCmdk();
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    cmdkActiveIndex = Math.max(cmdkActiveIndex - 1, 0);
    renderCmdk();
  } else if (e.key === 'Enter') {
    e.preventDefault();
    runCmdk(cmdkActiveIndex);
  }
});

document.addEventListener('keydown', (e) => {
  const isMod = e.metaKey || e.ctrlKey;
  if (isMod && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    cmdk.hidden ? openCmdk() : closeCmdk();
  } else if (e.key === 'Escape' && !cmdk.hidden) {
    closeCmdk();
  }
});

// Reveal on scroll
const revealTargets = document.querySelectorAll('.card, .featured-project, .contact-item, .about-text, .skills');
revealTargets.forEach((el) => el.classList.add('reveal'));

document.querySelectorAll('.project-grid .card').forEach((card, i) => {
  card.style.transitionDelay = Math.min(i, 5) * 70 + 'ms';
});

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const el = entry.target;
        el.classList.add('is-visible');
        el.addEventListener('transitionend', () => { el.style.transitionDelay = ''; }, { once: true });
        observer.unobserve(el);
      }
    });
  },
  { threshold: 0.15 }
);

revealTargets.forEach((el) => observer.observe(el));
