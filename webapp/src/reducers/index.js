import { combineReducers } from 'redux'

const tumor = (state = {tumorType:'', survivalRate:''}, action) => {
  switch(action.type) {
    case 'SELECT_REGION':
      return {
        ...state,
        region: action.payload.region,
        regionTitle: action.payload.regionTitle,
      }
    case 'RECEIVED_RESULTS':
      return {
        ...state,
        tumorType: action.payload.data.tumor_type,
        survivalRate: action.payload.data.survival_rate,
      }
    case 'STARTED_UPLOAD':
      return {
        tumorType: '',
        survivalRate: '',
      }
    default:
      return {
        ...state,
      }
  }
}

const images = (state = {}, action) => {
  switch(action.type) {
    case 'RECEIVED_RESULTS':
      return {
        ...state,
        results: action.payload.data.tumor,
      }
    case 'STARTED_UPLOAD':
      return {}
    default:
      return {
        ...state,
      }
  }
}

const status = (state = {loading: false}, action) => {
  switch(action.type) {
    case 'STARTED_UPLOAD':
      return {
        loading: true,
      }
    case 'FINISHED_UPLOAD':
      return {
        loading: false,
      }
    default:
      return {
        ...state,
      }
  }
}

const rootReducer = combineReducers({
  tumor,
  images,
  status,
})

export default rootReducer
