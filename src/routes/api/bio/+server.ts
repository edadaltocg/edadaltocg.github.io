import { json } from '@sveltejs/kit';
import bio from './bio.json';

export const GET = async () => {
	return json(bio);
};