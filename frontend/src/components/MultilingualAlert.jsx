import React, { useState } from 'react';
import { Globe, MessageSquare, Phone, Send } from 'lucide-react';
import { sendMultilingualAlert } from '../services/api';
import toast from 'react-hot-toast';

const MultilingualAlert = ({ onClose, initialData = {} }) => {
  const [formData, setFormData] = useState({
    message: initialData.message || '',
    location: initialData.location || '',
    severity: initialData.severity || 'medium',
    languages: ['en', 'hi', 'te'],
    phoneNumbers: initialData.phoneNumbers || [],
    ...initialData
  });

  const [isLoading, setIsLoading] = useState(false);

  const languageOptions = [
    { code: 'en', name: 'English', flag: '🇮🇳' },
    { code: 'hi', name: 'हिंदी (Hindi)', flag: '🇮🇳' },
    { code: 'bn', name: 'বাংলা (Bengali)', flag: '🇮🇳' },
    { code: 'as', name: 'অসমীয়া (Assamese)', flag: '🇮🇳' },
    { code: 'ne', name: 'नेपाली (Nepali)', flag: '🇳🇵' },
    { code: 'brx', name: 'बड़ो (Bodo)', flag: '🇮🇳' },
    { code: 'lus', name: 'Mizo (Mizo)', flag: '🇮🇳' },
    { code: 'mni', name: 'Meitei/Manipuri', flag: '🇮🇳' },
    { code: 'kha', name: 'Khasi', flag: '🇮🇳' },
    { code: 'grt', name: 'Garo', flag: '🇮🇳' }
  ];

  const handleLanguageToggle = (langCode) => {
    setFormData(prev => ({
      ...prev,
      languages: prev.languages.includes(langCode)
        ? prev.languages.filter(l => l !== langCode)
        : [...prev.languages, langCode]
    }));
  };

  const handlePhoneAdd = () => {
    const phone = prompt('Enter phone number:');
    if (phone && phone.trim()) {
      setFormData(prev => ({
        ...prev,
        phoneNumbers: [...prev.phoneNumbers, phone.trim()]
      }));
    }
  };

  const handlePhoneRemove = (index) => {
    setFormData(prev => ({
      ...prev,
      phoneNumbers: prev.phoneNumbers.filter((_, i) => i !== index)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.message.trim()) {
      toast.error('Please enter a message');
      return;
    }

    if (formData.languages.length === 0) {
      toast.error('Please select at least one language');
      return;
    }

    setIsLoading(true);

    try {
      const result = await sendMultilingualAlert({
        message: formData.message,
        location: formData.location,
        severity: formData.severity,
        languages: formData.languages,
        phoneNumbers: formData.phoneNumbers,
        coordinates: formData.coordinates || { lat: 17.3850, lng: 78.4867 }
      });

      if (result.success) {
        toast.success(`Alert sent in ${result.languages_sent.length} languages!`);
        onClose();
      } else {
        toast.error('Failed to send alert');
      }
    } catch (error) {
      toast.error('Error sending alert: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '1rem'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        padding: '2rem',
        maxWidth: '600px',
        width: '100%',
        maxHeight: '90vh',
        overflowY: 'auto',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '1.5rem'
        }}>
          <h2 style={{
            margin: 0,
            fontSize: '1.5rem',
            fontWeight: '700',
            color: '#1F2937',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <Globe size={24} color="#3B82F6" />
            Multilingual Emergency Alert
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer',
              color: '#6B7280'
            }}
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: '#374151',
              fontWeight: '500'
            }}>
              Alert Message *
            </label>
            <textarea
              value={formData.message}
              onChange={(e) => setFormData(prev => ({ ...prev, message: e.target.value }))}
              placeholder="Enter the emergency message..."
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '8px',
                border: '1px solid #D1D5DB',
                fontSize: '14px',
                resize: 'vertical',
                minHeight: '100px'
              }}
              required
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: '#374151',
              fontWeight: '500'
            }}>
              Location
            </label>
            <input
              type="text"
              value={formData.location}
              onChange={(e) => setFormData(prev => ({ ...prev, location: e.target.value }))}
              placeholder="Enter location..."
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '8px',
                border: '1px solid #D1D5DB',
                fontSize: '14px'
              }}
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: '#374151',
              fontWeight: '500'
            }}>
              Severity
            </label>
            <select
              value={formData.severity}
              onChange={(e) => setFormData(prev => ({ ...prev, severity: e.target.value }))}
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '8px',
                border: '1px solid #D1D5DB',
                fontSize: '14px'
              }}
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: '#374151',
              fontWeight: '500'
            }}>
              Languages *
            </label>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: '0.5rem'
            }}>
              {languageOptions.map(lang => (
                <label
                  key={lang.code}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.5rem',
                    border: '1px solid #D1D5DB',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    background: formData.languages.includes(lang.code) ? '#E0E7FF' : 'white',
                    transition: 'all 0.2s'
                  }}
                >
                  <input
                    type="checkbox"
                    checked={formData.languages.includes(lang.code)}
                    onChange={() => handleLanguageToggle(lang.code)}
                    style={{ margin: 0 }}
                  />
                  <span style={{ fontSize: '1.2rem' }}>{lang.flag}</span>
                  <span style={{ fontSize: '0.875rem' }}>{lang.name}</span>
                </label>
              ))}
            </div>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: '#374151',
              fontWeight: '500'
            }}>
              Phone Numbers (Optional)
            </label>
            <div style={{ marginBottom: '0.5rem' }}>
              <button
                type="button"
                onClick={handlePhoneAdd}
                style={{
                  background: '#3B82F6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '0.5rem 1rem',
                  fontSize: '0.875rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <Phone size={16} />
                Add Phone Number
              </button>
            </div>
            {formData.phoneNumbers.map((phone, index) => (
              <div
                key={index}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem',
                  background: '#F9FAFB',
                  borderRadius: '6px',
                  marginBottom: '0.5rem'
                }}
              >
                <span style={{ fontSize: '0.875rem' }}>{phone}</span>
                <button
                  type="button"
                  onClick={() => handlePhoneRemove(index)}
                  style={{
                    background: '#EF4444',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    cursor: 'pointer'
                  }}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>

          <div style={{
            display: 'flex',
            gap: '1rem',
            justifyContent: 'flex-end'
          }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                background: 'transparent',
                color: '#6B7280',
                border: '1px solid #D1D5DB',
                borderRadius: '8px',
                padding: '0.75rem 1.5rem',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              style={{
                background: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '0.75rem 1.5rem',
                fontSize: '14px',
                fontWeight: '600',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                opacity: isLoading ? 0.7 : 1,
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              <Send size={16} />
              {isLoading ? 'Sending...' : 'Send Alert'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default MultilingualAlert;
