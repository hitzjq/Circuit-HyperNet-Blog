(() => {
  const dialog = document.querySelector('#figure-dialog');
  const figures = [...document.querySelectorAll('.story-figure')];
  const scroll = document.querySelector('.figure-scroll');
  const zoom = document.querySelector('#figure-zoom');
  const close = document.querySelector('#figure-close');
  let lastOpener = null;
  if (dialog && typeof dialog.showModal === 'function') {
    const open = (event, figure) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      lastOpener = event.currentTarget;
      const image = figure.querySelector('img');
      const dialogImage = scroll.querySelector('img');
      for (const attribute of ['src', 'alt', 'width', 'height']) {
        dialogImage.setAttribute(attribute, image.getAttribute(attribute));
      }
      scroll.style.setProperty('--figure-native-width', image.getAttribute('width') + 'px');
      document.querySelector('#figure-dialog-title').textContent = figure.querySelector('.figure-toolbar > span').textContent;
      scroll.classList.remove('actual-size');
      zoom.setAttribute('aria-pressed', 'false');
      zoom.textContent = 'Actual size';
      dialog.showModal();
      scroll.scrollTo(0, 0);
      document.body.classList.add('dialog-open');
    };
    figures.forEach(figure => {
      figure.querySelectorAll('.figure-open, .figure-fallback').forEach(trigger => {
        trigger.addEventListener('click', event => open(event, figure));
      });
    });
    close.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (event) => {
      if (event.target === dialog) {
        const box = dialog.getBoundingClientRect();
        if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
      }
    });
    dialog.addEventListener('close', () => {
      document.body.classList.remove('dialog-open');
      if (lastOpener) lastOpener.focus();
    });
    zoom.addEventListener('click', () => {
      const actualSize = scroll.classList.toggle('actual-size');
      zoom.setAttribute('aria-pressed', String(actualSize));
      zoom.textContent = actualSize ? 'Fit to window' : 'Actual size';
    });
  } else {
    figures.forEach(figure => {
      const opener = figure.querySelector('.figure-open');
      const fallback = figure.querySelector('.figure-fallback');
      opener.addEventListener('click', () => { window.open(fallback.href, '_blank', 'noopener'); });
    });
  }
  const anchors = [...document.querySelectorAll('.contents a[href^="#"]')];
  const headings = anchors.map(link => document.querySelector(link.getAttribute('href')));
  if (headings.length) {
    let pending = false;
    const update = () => {
      let index = 0;
      headings.forEach((heading, i) => { if (heading.getBoundingClientRect().top < 180) index = i; });
      if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 5) index = headings.length - 1;
      anchors.forEach((link, i) => {
        link.classList.toggle('active', index === i);
        if (index === i) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
      pending = false;
    };
    window.addEventListener('scroll', () => { if (!pending) { pending = true; requestAnimationFrame(update); } }, { passive: true });
    update();
  }
  const mobileContents = document.querySelector('.mobile-contents');
  if (mobileContents) {
    mobileContents.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => { mobileContents.open = false; });
    });
  }
})();
