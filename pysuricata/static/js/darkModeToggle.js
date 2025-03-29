function toggleDarkMode() {
    const body = document.body;
    const icon = document.getElementById('toggle-icon');
  
    // Toggle the "dark" class on <body>
    body.classList.toggle('dark');
  
    // Update the icon based on the new state
    if (body.classList.contains('dark')) {
      // If we're now in dark mode, show a sun (☀️) to switch to light mode
      icon.textContent = '☀️';
    } else {
      // If we're now in light mode, show a moon (🌙) to switch to dark mode
      icon.textContent = '🌙';
    }
  }
  