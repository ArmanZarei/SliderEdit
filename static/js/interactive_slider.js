/**
 * Interactive Slider Gallery
 * Handles dynamic image loading and slider controls for GSTLoRA samples
 */

class InteractiveSliderGallery {
  constructor(containerId, samples) {
    this.container = document.getElementById(containerId);
    this.samples = samples;
    this.currentPage = 0;
    this.samplesPerPage = 3;
    this.imageCache = new Map();
    
    this.init();
  }
  
  init() {
    this.preloadImages();
    this.render();
    this.attachEventListeners();
  }
  
  preloadImages() {
    // Preload all images for smooth transitions
    this.samples.forEach(sample => {
      const img = new Image();
      img.src = sample.imagePath;
      this.imageCache.set(sample.imagePath, img);
    });
  }
  
  render() {
    const totalPages = Math.ceil(this.samples.length / this.samplesPerPage);
    
    const html = `
    <button class="gallery-nav-button prev" id="prevPage" ${this.currentPage === 0 ? 'disabled' : ''}>
        <i class="fas fa-chevron-left"></i>
    </button>

      <div class="slider-gallery-container">
        
        <div class="slider-gallery-wrapper" id="galleryWrapper">
          ${this.renderPages()}
        </div>
        
      </div>

    <button class="gallery-nav-button next" id="nextPage" ${this.currentPage >= totalPages - 1 ? 'disabled' : ''}>
        <i class="fas fa-chevron-right"></i>
    </button>
      
      <div class="gallery-pagination" id="galleryPagination">
        ${this.renderPagination(totalPages)}
      </div>
    `;
    
    this.container.innerHTML = html;
  }
  
  renderPages() {
    const totalPages = Math.ceil(this.samples.length / this.samplesPerPage);
    let pagesHtml = '';
    
    for (let page = 0; page < totalPages; page++) {
      const start = page * this.samplesPerPage;
      const end = Math.min(start + this.samplesPerPage, this.samples.length);
      const pageSamples = this.samples.slice(start, end);
      
      pagesHtml += `
        <div class="slider-gallery-page">
          ${pageSamples.map((sample, idx) => this.renderSliderItem(sample, start + idx)).join('')}
        </div>
      `;
    }
    
    return pagesHtml;
  }
  
  renderSliderItem(sample, index) {
    return `
      <div class="slider-contrabble-item">
        <div class="slider-image-container">
          <div class="slider-image-frame" id="frame_${index}"></div>
        </div>
        <div class="slider-prompt">${sample.prompt}</div>
        <div class="slider-controls">
          <div class="slider-control-wrapper">
            <span class="slider-control-label">Strength:</span>
            <button class="slider-step-button decrement" id="decrement_${index}" data-sample-index="${index}" data-action="decrement" disabled>
              <i class="fas fa-minus"></i>
            </button>
            <input 
              type="range" 
              class="slider-input" 
              id="slider_${index}"
              min="0" 
              max="${sample.numFrames - 1}" 
              value="0" 
              step="1"
              data-sample-index="${index}"
            />
            <button class="slider-step-button increment" id="increment_${index}" data-sample-index="${index}" data-action="increment">
              <i class="fas fa-plus"></i>
            </button>
          </div>
        </div>
      </div>
    `;
  }
  
  renderPagination(totalPages) {
    let html = '';
    for (let i = 0; i < totalPages; i++) {
      html += `
        <button 
          class="gallery-pagination-dot ${i === this.currentPage ? 'active' : ''}" 
          data-page="${i}"
        ></button>
      `;
    }
    return html;
  }
  
  attachEventListeners() {
    // Navigation buttons
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    
    if (prevBtn) {
      prevBtn.addEventListener('click', () => this.goToPage(this.currentPage - 1));
    }
    
    if (nextBtn) {
      nextBtn.addEventListener('click', () => this.goToPage(this.currentPage + 1));
    }
    
    // Pagination dots
    const pagination = document.getElementById('galleryPagination');
    if (pagination) {
      pagination.addEventListener('click', (e) => {
        if (e.target.classList.contains('gallery-pagination-dot')) {
          const page = parseInt(e.target.dataset.page);
          this.goToPage(page);
        }
      });
    }
    
    // Slider inputs and step buttons
    this.samples.forEach((sample, index) => {
      const slider = document.getElementById(`slider_${index}`);
      const decrementBtn = document.getElementById(`decrement_${index}`);
      const incrementBtn = document.getElementById(`increment_${index}`);
      
      if (slider) {
        slider.addEventListener('input', (e) => this.handleSliderChange(e, index));
        // Initialize the image
        this.updateImage(index, 0);
      }
      
      if (decrementBtn) {
        decrementBtn.addEventListener('click', () => this.stepSlider(index, -1));
      }
      
      if (incrementBtn) {
        incrementBtn.addEventListener('click', () => this.stepSlider(index, 1));
      }
    });
  }
  
  stepSlider(sampleIndex, direction) {
    const slider = document.getElementById(`slider_${sampleIndex}`);
    if (!slider) return;
    
    const currentValue = parseInt(slider.value);
    const newValue = currentValue + direction;
    const sample = this.samples[sampleIndex];
    
    if (newValue >= 0 && newValue < sample.numFrames) {
      slider.value = newValue;
      this.handleSliderChange({ target: slider }, sampleIndex);
    }
  }
  
  handleSliderChange(event, sampleIndex) {
    const frameIndex = parseInt(event.target.value);
    const sample = this.samples[sampleIndex];
    
    // Update button states
    const decrementBtn = document.getElementById(`decrement_${sampleIndex}`);
    const incrementBtn = document.getElementById(`increment_${sampleIndex}`);
    
    if (decrementBtn) {
      decrementBtn.disabled = frameIndex === 0;
    }
    
    if (incrementBtn) {
      incrementBtn.disabled = frameIndex === sample.numFrames - 1;
    }
    
    // Update slider background gradient
    const percentage = (frameIndex / (sample.numFrames - 1)) * 100;
    event.target.style.background = `linear-gradient(to right, #007bff 0%, #007bff ${percentage}%, #e9ecef ${percentage}%, #e9ecef 100%)`;
    
    // Update image
    this.updateImage(sampleIndex, frameIndex);
  }
  
  updateImage(sampleIndex, frameIndex) {
    const sample = this.samples[sampleIndex];
    const frameDiv = document.getElementById(`frame_${sampleIndex}`);
    
    if (!frameDiv) return;
    
    const img = this.imageCache.get(sample.imagePath);
    
    if (!img || !img.complete) {
      // Image not loaded yet, try again after a short delay
      setTimeout(() => this.updateImage(sampleIndex, frameIndex), 100);
      return;
    }
    
    // Get the actual container size
    const containerWidth = frameDiv.offsetWidth;
    const containerHeight = frameDiv.offsetHeight;
    
    // Original frame dimensions
    const frameWidth = 512;
    const frameHeight = 512;
    
    // Calculate the scale factor based on container size
    const scale = containerHeight / frameHeight;
    
    // Calculate scaled dimensions
    const scaledFrameWidth = frameWidth * scale;
    const scaledTotalWidth = (frameWidth * sample.numFrames) * scale;
    
    // Calculate position as percentage
    const xPosition = -(frameIndex * scaledFrameWidth);
    
    // Set background image and position
    frameDiv.style.backgroundImage = `url('${sample.imagePath}')`;
    frameDiv.style.backgroundPosition = `${xPosition}px 0`;
    frameDiv.style.backgroundSize = `${scaledTotalWidth}px ${containerHeight}px`;
    frameDiv.style.backgroundRepeat = 'no-repeat';
  }
  
  goToPage(pageIndex) {
    const totalPages = Math.ceil(this.samples.length / this.samplesPerPage);
    
    if (pageIndex < 0 || pageIndex >= totalPages) return;
    
    this.currentPage = pageIndex;
    
    // Update wrapper transform
    const wrapper = document.getElementById('galleryWrapper');
    wrapper.style.transform = `translateX(-${pageIndex * 100}%)`;
    
    // Update navigation buttons
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    
    prevBtn.disabled = pageIndex === 0;
    nextBtn.disabled = pageIndex >= totalPages - 1;
    
    // Update pagination dots
    document.querySelectorAll('.gallery-pagination-dot').forEach((dot, idx) => {
      dot.classList.toggle('active', idx === pageIndex);
    });
  }
}

// Initialize the gallery when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  // Sample data configuration
  // Each sample should have: imagePath, prompt, and numFrames
  const samples = [
    {
      imagePath: './static/images/gstlora_1d_samples/sample1.jpg',
      prompt: 'Turn on the light',
      numFrames: 23  // Adjust based on your actual image
    },
    {
      imagePath: './static/images/gstlora_1d_samples/sample5.jpg',
      prompt: 'Make her hair curly',
      numFrames: 28
    },
    {
      imagePath: './static/images/gstlora_1d_samples/sample2_new.jpg',
      prompt: 'Upgrade the car\'s design to a sports car',
      numFrames: 30
    },
    {
      imagePath: './static/images/gstlora_1d_samples/sample6.jpg',
      prompt: 'Make his hair white',
      numFrames: 38
    },
    {
      imagePath: './static/images/gstlora_1d_samples/sample3.jpg',
      prompt: 'Change the font style to dripping ink',
      numFrames: 24
    },
    {
      imagePath: './static/images/gstlora_1d_samples/sample4.jpg',
      prompt: 'Apply makeup to him',
      numFrames: 43
    },
    {
      imagePath: './static/images/gstlora_1d_samples/sample7.jpg',
      prompt: 'Make him laugh',
      numFrames: 37
    },
    {
      imagePath: './static/images/gstlora_1d_samples/sample8.jpg',
      prompt: 'Make the dragon exhale fire',
      numFrames: 26
    },
    {
      imagePath: './static/images/gstlora_1d_samples/sample9.jpg',
      prompt: 'Make the ninja golden',
      numFrames: 35
    }
  ];
  
  // Initialize the gallery
  new InteractiveSliderGallery('interactiveSliderGallery', samples);
});
