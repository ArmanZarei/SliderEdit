/**
 * 2D Slider Component
 * Handles dual-axis control for multi-instruction image editing
 */

class Slider2D {
  constructor(containerId, samples) {
    this.container = document.getElementById(containerId);
    this.samples = samples;
    this.imageCache = new Map();
    
    this.init();
  }
  
  init() {
    this.preloadImages();
    this.render();
    this.attachEventListeners();
  }
  
  preloadImages() {
    this.samples.forEach(sample => {
      const img = new Image();
      img.src = sample.imagePath;
      this.imageCache.set(sample.imagePath, img);
    });
  }
  
  render() {
    const html = `
      <div class="slider-2d-container">
        ${this.samples.map((sample, idx) => this.renderSliderItem(sample, idx)).join('')}
      </div>
    `;
    
    this.container.innerHTML = html;
  }
  
  renderSliderItem(sample, index) {
    // Create prompt with colored subprompts using full text
    const coloredPrompt = sample.instructions.map(instr => 
      `<span style="color: ${instr.color};">${instr.fullText}</span>`
    ).join(' and ');
    
    return `
      <div class="slider-2d-item">
        <div class="slider-2d-image-wrapper">
          <div class="slider-2d-image-frame" id="frame2d_${index}"></div>
        </div>
        <div class="slider-2d-prompt">"${coloredPrompt}"</div>
        <div class="slider-2d-controls">
          ${sample.instructions.map((instruction, instrIdx) => 
            this.renderControlGroup(instruction, index, instrIdx)
          ).join('')}
        </div>
      </div>
    `;
  }
  
  renderControlGroup(instruction, sampleIndex, instrIndex) {
    const sliderId = `slider2d_${sampleIndex}_${instrIndex}`;
    const decrementId = `decrement2d_${sampleIndex}_${instrIndex}`;
    const incrementId = `increment2d_${sampleIndex}_${instrIndex}`;
    
    return `
      <div class="slider-2d-control-wrapper">
        <span class="slider-2d-control-label" style="color: ${instruction.color};">${instruction.text}:</span>
        <button 
          class="slider-2d-step-button decrement" 
          id="${decrementId}"
          data-sample-index="${sampleIndex}"
          data-instr-index="${instrIndex}"
          data-action="decrement"
          disabled
        >
          <i class="fas fa-minus"></i>
        </button>
        <input 
          type="range" 
          class="slider-2d-input" 
          id="${sliderId}"
          min="0" 
          max="6" 
          value="0" 
          step="1"
          data-sample-index="${sampleIndex}"
          data-instr-index="${instrIndex}"
          data-color="${instruction.color}"
        />
        <button 
          class="slider-2d-step-button increment" 
          id="${incrementId}"
          data-sample-index="${sampleIndex}"
          data-instr-index="${instrIndex}"
          data-action="increment"
        >
          <i class="fas fa-plus"></i>
        </button>
      </div>
    `;
  }
  
  attachEventListeners() {
    this.samples.forEach((sample, sampleIndex) => {
      sample.instructions.forEach((instruction, instrIndex) => {
        const slider = document.getElementById(`slider2d_${sampleIndex}_${instrIndex}`);
        const decrementBtn = document.getElementById(`decrement2d_${sampleIndex}_${instrIndex}`);
        const incrementBtn = document.getElementById(`increment2d_${sampleIndex}_${instrIndex}`);
        
        if (slider) {
          slider.addEventListener('input', (e) => this.handleSliderChange(e, sampleIndex, instrIndex));
          // Initialize the slider color and style
          this.initializeSliderStyle(sampleIndex, instrIndex);
          // Initialize the image
          this.updateImage(sampleIndex);
        }
        
        if (decrementBtn) {
          decrementBtn.addEventListener('click', () => this.stepSlider(sampleIndex, instrIndex, -1));
        }
        
        if (incrementBtn) {
          incrementBtn.addEventListener('click', () => this.stepSlider(sampleIndex, instrIndex, 1));
        }
      });
    });
  }
  
  initializeSliderStyle(sampleIndex, instrIndex) {
    const slider = document.getElementById(`slider2d_${sampleIndex}_${instrIndex}`);
    if (!slider) return;
    
    const color = slider.dataset.color;
    const value = parseInt(slider.value);
    const percentage = (value / 6) * 100;
    
    // Set initial background
    slider.style.background = `linear-gradient(to right, ${color} 0%, ${color} ${percentage}%, #e9ecef ${percentage}%, #e9ecef 100%)`;
    
    // Create style for thumb color
    const style = document.createElement('style');
    style.id = `slider-style-${sampleIndex}-${instrIndex}`;
    style.textContent = `
      #slider2d_${sampleIndex}_${instrIndex}::-webkit-slider-thumb {
        background: ${color};
        box-shadow: 0 2px 8px ${color}66;
      }
      #slider2d_${sampleIndex}_${instrIndex}::-moz-range-thumb {
        background: ${color};
        box-shadow: 0 2px 8px ${color}66;
      }
    `;
    document.head.appendChild(style);
  }
  
  stepSlider(sampleIndex, instrIndex, direction) {
    const slider = document.getElementById(`slider2d_${sampleIndex}_${instrIndex}`);
    if (!slider) return;
    
    const currentValue = parseInt(slider.value);
    const newValue = currentValue + direction;
    
    if (newValue >= 0 && newValue <= 6) {
      slider.value = newValue;
      this.handleSliderChange({ target: slider }, sampleIndex, instrIndex);
    }
  }
  
  handleSliderChange(event, sampleIndex, instrIndex) {
    const value = parseInt(event.target.value);
    const color = event.target.dataset.color;
    
    // Update button states
    const decrementBtn = document.getElementById(`decrement2d_${sampleIndex}_${instrIndex}`);
    const incrementBtn = document.getElementById(`increment2d_${sampleIndex}_${instrIndex}`);
    
    if (decrementBtn) {
      decrementBtn.disabled = value === 0;
    }
    
    if (incrementBtn) {
      incrementBtn.disabled = value === 6;
    }
    
    // Update slider background gradient and thumb color
    const percentage = (value / 6) * 100;
    event.target.style.background = `linear-gradient(to right, ${color} 0%, ${color} ${percentage}%, #e9ecef ${percentage}%, #e9ecef 100%)`;
    
    // Update thumb color using CSS custom property
    const style = document.createElement('style');
    style.textContent = `
      #slider2d_${sampleIndex}_${instrIndex}::-webkit-slider-thumb {
        background: ${color};
        box-shadow: 0 2px 8px ${color}66;
      }
      #slider2d_${sampleIndex}_${instrIndex}::-moz-range-thumb {
        background: ${color};
        box-shadow: 0 2px 8px ${color}66;
      }
    `;
    if (!document.getElementById(`slider-style-${sampleIndex}-${instrIndex}`)) {
      style.id = `slider-style-${sampleIndex}-${instrIndex}`;
      document.head.appendChild(style);
    }
    
    // Update image
    this.updateImage(sampleIndex);
  }
  
  updateImage(sampleIndex) {
    const sample = this.samples[sampleIndex];
    const frameDiv = document.getElementById(`frame2d_${sampleIndex}`);
    
    if (!frameDiv) return;
    
    // Get slider values
    const slider1 = document.getElementById(`slider2d_${sampleIndex}_0`);
    const slider2 = document.getElementById(`slider2d_${sampleIndex}_1`);
    
    if (!slider1 || !slider2) return;
    
    const value1 = parseInt(slider1.value);
    const value2 = parseInt(slider2.value);
    
    // Calculate background position
    // Bottom-right is (0,0) -> backgroundPosition: 100% 100% (image grid starts at bottom-right)
    // Top-left is (6,6) -> backgroundPosition: 0% 0% (image grid ends at top-left)
    // We need to invert both axes since the grid is reversed
    
    const xPercent = 100 - (value1 / 6) * 100; // 100 to 0 (inverted)
    const yPercent = 100 - (value2 / 6) * 100; // 100 to 0 (inverted)
    
    const img = this.imageCache.get(sample.imagePath);
    
    if (!img || !img.complete) {
      setTimeout(() => this.updateImage(sampleIndex), 100);
      return;
    }
    
    frameDiv.style.backgroundImage = `url('${sample.imagePath}')`;
    frameDiv.style.backgroundPosition = `${xPercent}% ${yPercent}%`;
    frameDiv.style.backgroundSize = '700% 700%';
    frameDiv.style.backgroundRepeat = 'no-repeat';
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  const samples2D = [
    {
      imagePath: './static/images/gstlora_2d_samples/sample1.jpg',
      prompt: 'Make her hair curly and make her smile',
      instructions: [
        { text: 'Curliness', fullText: 'make her hair curly', color: '#FF6B6B' },
        { text: 'Smile', fullText: 'make her smile', color: '#4ECDC4' }
      ]
    },
    {
      imagePath: './static/images/gstlora_2d_samples/sample2.jpg',
      prompt: 'Make him old and make him laugh',
      instructions: [
        { text: 'Age', fullText: 'make him old', color: '#95E1D3' },
        { text: 'Laughter', fullText: 'make him laugh', color: '#F38181' }
      ]
    }
  ];
  
  new Slider2D('slider2dGallery', samples2D);
});
